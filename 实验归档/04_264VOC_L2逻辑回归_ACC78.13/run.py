import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA_PATH = REPO_ROOT / "data" / "voc_dataset_table1_1+2_vs_3_bpd.mat"
FEATURE_PATH = (
    REPO_ROOT
    / "实验归档"
    / "01_删除棉球物质_BPD_MLP_ACC78.12"
    / "stable_features.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "result" / "reproduced" / "04_logistic_regression"


def load_inputs():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"缺少私有数据：{DATA_PATH}\n请参阅 data/README.md。")
    data = loadmat(DATA_PATH)
    features = pd.read_csv(FEATURE_PATH)
    x = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64).reshape(-1)
    if x.shape != (159, 1692) or y.shape != (159,):
        raise ValueError(f"数据形状不符：X={x.shape}, y={y.shape}")
    indices = features["VOC_Index"].to_numpy(dtype=int)
    if len(indices) != 264 or len(np.unique(indices)) != 264:
        raise ValueError(f"稳定 VOC 应为 264 个，实际为 {len(np.unique(indices))} 个。")
    return x[:, indices], y


def main() -> None:
    parser = argparse.ArgumentParser(description="复现 264 VOC 的 L2 逻辑回归实验")
    parser.add_argument("--check", action="store_true", help="只检查数据与配置")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    x, y = load_inputs()
    all_indices = np.arange(len(y))
    development_indices, _ = train_test_split(
        all_indices, test_size=0.20, random_state=42, stratify=y
    )
    train_indices, val_indices = train_test_split(
        development_indices,
        test_size=0.25,
        random_state=1042,
        stratify=y[development_indices],
    )
    print(f"输入形状：{x.shape}；内部训练/验证：{len(train_indices)}/{len(val_indices)}")
    if args.check:
        print("数据与复现配置检查通过。")
        return

    results = []
    for c_value in (0.001, 0.01, 0.1, 1, 10):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=c_value,
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            ),
        )
        model.fit(x[train_indices], y[train_indices])
        probabilities = model.predict_proba(x[val_indices])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y[val_indices], predictions, labels=[0, 1]
        ).ravel()
        results.append(
            {
                "C": c_value,
                "Accuracy": accuracy_score(y[val_indices], predictions),
                "Sensitivity": tp / max(tp + fn, 1),
                "Specificity": tn / max(tn + fp, 1),
                "F1": f1_score(y[val_indices], predictions, zero_division=0),
                "AUC": roc_auc_score(y[val_indices], probabilities),
            }
        )

    result = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "logistic_regression_264.csv"
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False))
    print(f"复现结果已保存：{output_path}")


if __name__ == "__main__":
    main()
