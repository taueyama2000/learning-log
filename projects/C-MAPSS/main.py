"""VS Codeとターミナルから実行するC-MAPSSパイプラインの入口。"""

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
# srcレイアウトのパッケージを、インストールせず実行できるようにします。
sys.path.insert(0, str(PROJECT_DIR / "src"))

from cmapss.pipeline import RunConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C-MAPSSのEDAとRUL予測を実行します。")
    parser.add_argument(
        "--dataset",
        choices=["FD001", "FD002", "FD003", "FD004"],
        default="FD001",
        help="使用するデータセット（初期値: FD001）",
    )
    parser.add_argument("--rul-cap", type=int, default=125, help="学習用RULの上限")
    parser.add_argument("--window", type=int, default=5, help="移動統計の窓幅")
    parser.add_argument("--trees", type=int, default=250, help="Random Forestの決定木数")
    parser.add_argument("--skip-eda", action="store_true", help="EDAの出力を省略")
    parser.add_argument("--no-save-model", action="store_true", help="モデル保存を省略")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        project_dir=PROJECT_DIR,
        dataset=args.dataset,
        rul_cap=args.rul_cap,
        rolling_window=args.window,
        trees=args.trees,
        skip_eda=args.skip_eda,
        save_model=not args.no_save_model,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
