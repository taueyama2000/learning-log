"""C-MAPSSデータの読み込みとRUL正解ラベルの作成。"""

from pathlib import Path

import numpy as np
import pandas as pd


SETTING_COLUMNS = [f"setting_{number}" for number in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{number}" for number in range(1, 22)]
COLUMNS = ["unit_id", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]


def load_split(data_dir: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train、test、testの正解RULを読み込む。"""
    dataset = dataset.upper()
    train = pd.read_csv(
        data_dir / f"train_{dataset}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )
    test = pd.read_csv(
        data_dir / f"test_{dataset}.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )
    test_rul = pd.read_csv(
        data_dir / f"RUL_{dataset}.txt",
        header=None,
        names=["RUL"],
    )

    # RULファイルは1行目がunit_id=1、2行目がunit_id=2、…の順です。
    test_rul.index = np.arange(1, len(test_rul) + 1)
    test_rul.index.name = "unit_id"
    return train, test, test_rul


def add_rul(data: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """trainの各行に、故障までの残りサイクル数を追加する。"""
    result = data.copy()
    final_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["RUL_raw"] = final_cycle - result["cycle"]
    result["RUL"] = result["RUL_raw"].clip(upper=cap) if cap is not None else result["RUL_raw"]
    return result
