"""RUL回帰モデルの評価指標。"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_score(y_true, y_predicted) -> float:
    """過大予測を強く罰するC-MAPSSのNASA score。小さいほど良い。"""
    difference = np.asarray(y_predicted) - np.asarray(y_true)
    penalty = np.where(
        difference < 0,
        np.exp(-difference / 13) - 1,
        np.exp(difference / 10) - 1,
    )
    return float(penalty.sum())


def regression_metrics(y_true, y_predicted) -> dict[str, float]:
    """RMSE、MAE、R2、NASA scoreをまとめて計算する。"""
    return {
        "RMSE": mean_squared_error(y_true, y_predicted) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_predicted),
        "R2": r2_score(y_true, y_predicted),
        "NASA_score": nasa_score(y_true, y_predicted),
    }
