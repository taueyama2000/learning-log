"""データ処理と評価指標の小さな動作確認。"""

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from cmapss.data import add_rul
from cmapss.metrics import nasa_score


class AddRulTest(unittest.TestCase):
    def test_rul_counts_down_to_zero_for_each_engine(self):
        data = pd.DataFrame(
            {"unit_id": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]}
        )
        result = add_rul(data)
        self.assertEqual(result["RUL"].tolist(), [2, 1, 0, 1, 0])

    def test_rul_cap_limits_healthy_period(self):
        data = pd.DataFrame({"unit_id": [1, 1], "cycle": [1, 201]})
        result = add_rul(data, cap=125)
        self.assertEqual(result["RUL"].tolist(), [125, 0])


class NasaScoreTest(unittest.TestCase):
    def test_perfect_prediction_is_zero(self):
        self.assertEqual(nasa_score([10, 20], [10, 20]), 0.0)

    def test_over_prediction_has_larger_penalty(self):
        under = nasa_score([20], [10])
        over = nasa_score([20], [30])
        self.assertGreater(over, under)


if __name__ == "__main__":
    unittest.main()
