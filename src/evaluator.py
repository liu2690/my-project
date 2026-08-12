"""
测试评价模块

只使用 test 数据集
"""

import numpy as np
import torch

from src.metrics import compute_metrics


def evaluate_model(
        model,
        test_loader,
        threshold=0.5,
        device=None
):

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else
            "cpu"
        )

    model.to(device)
    model.eval()

    preds = []
    probs = []
    targets = []

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)

            output, _ = model(x)

            # class 1 probability
            probability = torch.softmax(
                output,
                dim=1
            )[:, 1]

            # 使用 Validation 阶段确定的 threshold
            pred = (
                probability >= threshold
            ).long()

            preds.extend(
                pred.detach()
                .cpu()
                .numpy()
            )

            probs.extend(
                probability.detach()
                .cpu()
                .numpy()
            )

            targets.extend(
                y.detach()
                .cpu()
                .numpy()
            )

    result = compute_metrics(
        targets,
        preds,
        probs
    )

    # 保存原始预测信息
    result["Targets"] = np.asarray(
        targets
    )

    result["Predictions"] = np.asarray(
        preds
    )

    result["Probabilities"] = np.asarray(
        probs
    )

    result["Threshold"] = float(
        threshold
    )

    return result