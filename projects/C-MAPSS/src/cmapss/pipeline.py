"""EDA、モデル比較、公式test評価を一続きで実行する。"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

# VS Codeタスクやターミナルから実行しても、画面待ちで止まらない描画方式です。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import SENSOR_COLUMNS, add_rul, load_split
from .features import add_rolling_features, feature_columns, select_active_sensors
from .metrics import regression_metrics


@dataclass(frozen=True)
class RunConfig:
    """main.pyから受け取る実行設定。"""

    project_dir: Path
    dataset: str = "FD001"
    rul_cap: int = 125
    rolling_window: int = 5
    random_state: int = 42
    trees: int = 250
    skip_eda: bool = False
    save_model: bool = True


def _save_eda(train: pd.DataFrame, test: pd.DataFrame, sensors: list[str], output_dir: Path) -> None:
    """EDAの表とグラフをartifactsへ保存する。"""
    train_life = train.groupby("unit_id")["cycle"].max().rename("train_lifetime")
    test_length = test.groupby("unit_id")["cycle"].max().rename("test_observed_cycles")

    quality = pd.DataFrame(
        {
            "rows": [len(train), len(test)],
            "units": [train["unit_id"].nunique(), test["unit_id"].nunique()],
            "missing_values": [int(train.isna().sum().sum()), int(test.isna().sum().sum())],
            "duplicate_rows": [int(train.duplicated().sum()), int(test.duplicated().sum())],
        },
        index=["train", "test"],
    )
    quality.to_csv(output_dir / "data_quality.csv")
    pd.concat([train_life.describe(), test_length.describe()], axis=1).to_csv(
        output_dir / "lifetime_summary.csv"
    )

    # 一定値列の相関は定義できないため、値が変化するセンサーだけ計算します。
    correlations = pd.Series(np.nan, index=SENSOR_COLUMNS, dtype=float)
    correlations.loc[sensors] = train[sensors].corrwith(train["RUL"])
    sensor_stats = pd.DataFrame(
        {
            "mean": train[SENSOR_COLUMNS].mean(),
            "std": train[SENSOR_COLUMNS].std(),
            "n_unique": train[SENSOR_COLUMNS].nunique(),
            "corr_with_RUL": correlations,
        }
    )
    sensor_stats["abs_corr_with_RUL"] = sensor_stats["corr_with_RUL"].abs()
    sensor_stats.sort_values("abs_corr_with_RUL", ascending=False).to_csv(
        output_dir / "sensor_statistics.csv"
    )

    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train_life, bins=18, kde=True, ax=axes[0], color="#2563eb")
    axes[0].set(title="Train engine lifetime", xlabel="cycles")
    sns.histplot(test_length, bins=18, kde=True, ax=axes[1], color="#f97316")
    axes[1].set(title="Observed test length", xlabel="cycles")
    figure.tight_layout()
    figure.savefig(output_dir / "lifetime_distribution.png", dpi=150)
    plt.close(figure)

    top = sensor_stats.loc[sensors].nlargest(10, "abs_corr_with_RUL")
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(data=top.reset_index(), x="corr_with_RUL", y="index", color="#2563eb", ax=axis)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(title="Top sensor correlations with capped RUL", xlabel="Pearson correlation", ylabel="")
    figure.tight_layout()
    figure.savefig(output_dir / "sensor_rul_correlation.png", dpi=150)
    plt.close(figure)


def _validation_endpoints(validation: pd.DataFrame) -> pd.DataFrame:
    """各検証エンジンの寿命50%、70%、90%時点を取り出す。"""
    parts = []
    for fraction in (0.5, 0.7, 0.9):
        lifetimes = validation.groupby("unit_id")["cycle"].max()
        cutoffs = np.floor(lifetimes * fraction).astype(int).clip(lower=1)
        keys = pd.MultiIndex.from_arrays([cutoffs.index, cutoffs.values])
        mask = validation.set_index(["unit_id", "cycle"]).index.isin(keys)
        part = validation.loc[mask].copy()
        part["cutoff_fraction"] = fraction
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _candidate_models(config: RunConfig):
    """比較する3モデルを用意する。"""
    return {
        "Dummy median": DummyRegressor(strategy="median"),
        "Ridge": Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=config.trees,
            min_samples_leaf=3,
            max_features=0.7,
            n_jobs=-1,
            random_state=config.random_state,
        ),
    }


def run_pipeline(config: RunConfig) -> pd.DataFrame:
    """EDAから公式test評価まで実行し、最終評価指標を返す。"""
    dataset = config.dataset.upper()
    data_dir = config.project_dir / "CMAPSSData"
    output_dir = config.project_dir / "artifacts" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] {dataset}を読み込んでいます...")
    train, test, test_rul = load_split(data_dir, dataset)
    train = add_rul(train, cap=config.rul_cap)
    sensors = select_active_sensors(train)
    print(f"      train={len(train):,}行, test={len(test):,}行, 使用センサー={len(sensors)}個")

    if not config.skip_eda:
        print("[2/5] EDAの表とグラフを保存しています...")
        _save_eda(train, test, sensors, output_dir)
    else:
        print("[2/5] --skip-edaによりEDAを省略しました。")

    print("[3/5] 時系列特徴量を作成しています...")
    train_features = add_rolling_features(train, sensors, config.rolling_window)
    test_features = add_rolling_features(test, sensors, config.rolling_window)
    columns = feature_columns(train_features, sensors, config.rolling_window)

    random = np.random.default_rng(config.random_state)
    units = train_features["unit_id"].unique()
    random.shuffle(units)
    split_at = int(len(units) * 0.8)
    fit_rows = train_features[train_features["unit_id"].isin(units[:split_at])]
    validation = train_features[train_features["unit_id"].isin(units[split_at:])]
    endpoints = _validation_endpoints(validation)

    print("[4/5] 3モデルを学習・比較しています...")
    models = _candidate_models(config)
    validation_rows = []
    for name, model in models.items():
        model.fit(fit_rows[columns], fit_rows["RUL"])
        predicted = np.clip(model.predict(endpoints[columns]), 0, config.rul_cap)
        validation_rows.append(
            {"model": name, **regression_metrics(endpoints["RUL_raw"], predicted)}
        )
    validation_metrics = pd.DataFrame(validation_rows).sort_values("RMSE")
    validation_metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    print(validation_metrics.round(3).to_string(index=False))

    best_name = validation_metrics.iloc[0]["model"]
    best_model = models[best_name]
    print(f"[5/5] 最良モデル（{best_name}）を全trainで再学習し、公式testを評価します...")
    best_model.fit(train_features[columns], train_features["RUL"])

    last_indices = test_features.groupby("unit_id")["cycle"].idxmax()
    test_endpoints = test_features.loc[last_indices].sort_values("unit_id")
    if test_endpoints["unit_id"].tolist() != test_rul.index.tolist():
        raise ValueError("testエンジンIDとRULファイルの行順が一致しません。")

    predicted = np.clip(best_model.predict(test_endpoints[columns]), 0, config.rul_cap)
    truth = test_rul["RUL"].to_numpy()
    final_metrics = pd.DataFrame(
        [{"dataset": dataset, "model": best_name, **regression_metrics(truth, predicted)}]
    )
    predictions = pd.DataFrame(
        {
            "unit_id": test_endpoints["unit_id"].to_numpy(),
            "last_observed_cycle": test_endpoints["cycle"].to_numpy(),
            "RUL_true": truth,
            "RUL_pred": predicted,
            "error": predicted - truth,
        }
    )
    final_metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(truth, predicted, alpha=0.7, color="#2563eb")
    limit = max(truth.max(), predicted.max()) + 5
    axis.plot([0, limit], [0, limit], "--", color="black", linewidth=1)
    axis.set(xlabel="True RUL", ylabel="Predicted RUL", title=f"{dataset} official test")
    figure.tight_layout()
    figure.savefig(output_dir / "test_predictions.png", dpi=150)
    plt.close(figure)

    if config.save_model:
        joblib.dump(
            {
                "model": best_model,
                "feature_columns": columns,
                "active_sensors": sensors,
                "rul_cap": config.rul_cap,
            },
            output_dir / "baseline_model.joblib",
            compress=3,
        )

    print("\n公式test結果")
    print(final_metrics.round(3).to_string(index=False))
    print(f"\n成果物: {output_dir}")
    return final_metrics
