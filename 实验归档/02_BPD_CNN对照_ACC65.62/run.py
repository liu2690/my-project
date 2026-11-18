import argparse
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
MLP_DIR = REPO_ROOT / "实验归档" / "01_删除棉球物质_BPD_MLP_ACC78.12"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_notebook_cells import execute_cells


def main() -> None:
    parser = argparse.ArgumentParser(description="复现同数据 BPD-CNN 对照实验")
    parser.add_argument("--check", action="store_true", help="只检查数据与代码")
    args = parser.parse_args()

    execute_cells(
        repo_root=REPO_ROOT,
        notebook_path=EXPERIMENT_DIR / "testing_CNN.ipynb",
        cell_indices=[0, 14],
        output_path=REPO_ROOT / "result" / "reproduced" / "02_cnn" / "executed.ipynb",
        import_paths=[MLP_DIR],
        module_override=EXPERIMENT_DIR / "modules_CNN.py",
        check_only=args.check,
    )


if __name__ == "__main__":
    main()
