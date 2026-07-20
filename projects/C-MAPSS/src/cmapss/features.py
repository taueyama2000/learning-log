"""センサー時系列から学習用の特徴量を作る。"""

import pandas as pd

from .data import SENSOR_COLUMNS, SETTING_COLUMNS


def select_active_sensors(train: pd.DataFrame, threshold: float = 1e-8) -> list[str]:
    """値がほぼ一定のセンサーを除外する。"""
    standard_deviation = train[SENSOR_COLUMNS].std()
    return standard_deviation.index[standard_deviation > threshold].tolist()


def add_rolling_features(
    data: pd.DataFrame,
    sensors: list[str],
    window: int = 5,
) -> pd.DataFrame:
    """直近windowサイクルの平均、標準偏差、前回との差を追加する。"""
    result = data.sort_values(["unit_id", "cycle"]).copy()
    grouped = result.groupby("unit_id", sort=False)

    for sensor in sensors:
        result[f"{sensor}_roll_mean_{window}"] = grouped[sensor].transform(
            lambda values: values.rolling(window, min_periods=1).mean()
        )
        result[f"{sensor}_roll_std_{window}"] = grouped[sensor].transform(
            lambda values: values.rolling(window, min_periods=1).std().fillna(0)
        )
        result[f"{sensor}_delta"] = grouped[sensor].diff().fillna(0)
    return result


def feature_columns(data: pd.DataFrame, sensors: list[str], window: int = 5) -> list[str]:
    """モデルへ入力する列名を返す。"""
    raw_columns = ["cycle", *SETTING_COLUMNS, *sensors]
    suffixes = (f"_roll_mean_{window}", f"_roll_std_{window}", "_delta")
    return [column for column in data.columns if column in raw_columns or column.endswith(suffixes)]
