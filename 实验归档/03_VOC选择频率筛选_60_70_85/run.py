import argparse
import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA_PATH = REPO_ROOT / "data" / "voc_dataset_table1_1+2_vs_3_bpd.mat"
MLP_MODULE_PATH = (
    REPO_ROOT / "实验归档" / "01_删除棉球物质_BPD_MLP_ACC78.12" / "modules.py"
)
DEFAULT_OUTPUT = REPO_ROOT / "result" / "reproduced" / "03_voc_thresholds"
DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mlp_class():
    spec = importlib.util.spec_from_file_location("bpd_mlp_modules", MLP_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模型：{MLP_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MultiView


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_and_load():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"缺少私有数据：{DATA_PATH}\n请参阅 data/README.md。")
    data = loadmat(DATA_PATH)
    samples = np.asarray(data["X"], dtype=np.float32)
    labels = np.asarray(data["y"], dtype=np.int64).reshape(-1)
    if samples.shape != (159, 1692) or labels.shape != (159,):
        raise ValueError(f"数据形状不符：X={samples.shape}, y={labels.shape}")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("y 必须是 0/1 二分类标签。")
    return samples, labels


def build_masks(feature_table: pd.DataFrame) -> dict[str, np.ndarray]:
    required = {"VOC_Index", "Selection_Frequency"}
    if not required.issubset(feature_table.columns):
        raise ValueError(f"选择频率表缺少字段：{sorted(required - set(feature_table.columns))}")
    masks = {}
    for name, threshold in {"gt60": 0.60, "gt70": 0.70, "gt85": 0.85}.items():
        mask = np.zeros(len(feature_table), dtype=np.float32)
        selected = feature_table.loc[
            feature_table["Selection_Frequency"] > threshold, "VOC_Index"
        ].to_numpy(dtype=int)
        mask[selected] = 1.0
        masks[name] = mask
    return masks


def feature_cluster(samples: np.ndarray, seed: int) -> torch.Tensor:
    clusters = KMeans(n_clusters=3, n_init=10, random_state=seed).fit_predict(samples.T)
    return torch.nn.functional.one_hot(
        torch.tensor(clusters, dtype=torch.long), num_classes=3
    ).T


def train_and_evaluate(
    *,
    name: str,
    mask: np.ndarray,
    samples: torch.Tensor,
    labels: torch.Tensor,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    kmeans_mask: torch.Tensor,
    epochs: int,
    seed: int,
) -> dict[str, float | int | str]:
    set_seed(seed)
    model_class = load_mlp_class()
    model = model_class(
        k_mean_mask=kmeans_mask,
        num_blocks=4,
        head_dim=256,
        num_class=2,
        temperature=1.0,
    ).to(DEVICE, dtype=DTYPE)
    for parameter in model.logist.parameters():
        parameter.requires_grad = False

    optimizer = optim.Adam(model.mlp_classifier.parameters(), lr=5e-4, weight_decay=5e-3)
    criterion = nn.CrossEntropyLoss()
    stable_mask = torch.tensor(mask, dtype=DTYPE, device=DEVICE)
    train_loader = DataLoader(
        TensorDataset(samples[train_indices], labels[train_indices]),
        batch_size=16,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(samples[val_indices], labels[val_indices]),
        batch_size=16,
        shuffle=False,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE, dtype=DTYPE)
            y_batch = y_batch.to(DEVICE).long()
            optimizer.zero_grad()
            output = model.mlp_classifier(x_batch * stable_mask)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"{name} Epoch {epoch + 1}/{epochs}, Loss={total_loss / len(train_loader):.5f}")

    probabilities, targets = [], []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            output = model.mlp_classifier(x_batch.to(DEVICE, dtype=DTYPE) * stable_mask)
            probabilities.extend(torch.softmax(output.float(), dim=1)[:, 1].cpu().numpy())
            targets.extend(y_batch.numpy())

    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    return {
        "Panel": name,
        "VOC_Count": int(mask.sum()),
        "Accuracy": (tp + tn) / len(targets),
        "Sensitivity": tp / max(tp + fn, 1),
        "Specificity": tn / max(tn + fp, 1),
        "F1": 2 * tp / max(2 * tp + fp + fn, 1),
        "AUC": roc_auc_score(targets, probabilities),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="复现 VOC 选择频率 60/70/85% 筛选实验")
    parser.add_argument("--check", action="store_true", help="只检查数据与配置")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    samples_np, labels_np = validate_and_load()
    feature_table = pd.read_csv(EXPERIMENT_DIR / "feature_selection_frequency.csv")
    masks = build_masks(feature_table)
    all_indices = np.arange(len(labels_np))
    development_indices, _ = train_test_split(
        all_indices, test_size=0.20, random_state=42, stratify=labels_np
    )
    train_indices, val_indices = train_test_split(
        development_indices,
        test_size=0.25,
        random_state=1042,
        stratify=labels_np[development_indices],
    )
    print("面板特征数：", {name: int(mask.sum()) for name, mask in masks.items()})
    print(f"内部训练/验证样本数：{len(train_indices)}/{len(val_indices)}")
    if args.check:
        print("数据与复现配置检查通过。")
        return

    set_seed(42)
    samples = torch.tensor(samples_np, dtype=DTYPE)
    labels = torch.tensor(labels_np, dtype=torch.long)
    kmeans_mask = feature_cluster(samples_np[train_indices], seed=42)
    results = [
        train_and_evaluate(
            name=name,
            mask=masks[name],
            samples=samples,
            labels=labels,
            train_indices=train_indices,
            val_indices=val_indices,
            kmeans_mask=kmeans_mask,
            epochs=args.epochs,
            seed=42,
        )
        for name in ("gt60", "gt70", "gt85")
    ]
    summary = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "validation_summary.csv"
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"复现结果已保存：{output_path}")


if __name__ == "__main__":
    main()
