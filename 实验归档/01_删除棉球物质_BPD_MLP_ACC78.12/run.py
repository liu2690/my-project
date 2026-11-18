import argparse
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_notebook_cells import execute_cells


def main() -> None:
    parser = argparse.ArgumentParser(description="复现删除棉球物质后的 BPD-MLP 实验")
    parser.add_argument("--check", action="store_true", help="只检查数据与代码")
    args = parser.parse_args()

    execute_cells(
        repo_root=REPO_ROOT,
        notebook_path=EXPERIMENT_DIR / "testing_MLP.ipynb",
        cell_indices=[0, 1],
        output_path=REPO_ROOT / "result" / "reproduced" / "01_mlp" / "executed.ipynb",
        import_paths=[EXPERIMENT_DIR],
        check_only=args.check,
    )


if __name__ == "__main__":
    main()
