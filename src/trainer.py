"""
模型训练模块

负责:
1. train训练
2. val验证
3. 使用类别加权损失缓解类别不平衡
4. 使用Validation F1选择最佳模型
5. 保存最佳模型与最佳mask
"""

import copy

import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import f1_score



def train_model(
        model,
        train_loader,
        val_loader,
        epochs=30,
        lr=5e-4,
        weight_decay=5e-3,
        device=None
):

    """
    训练 MultiView + CNN 模型

    Parameters
    ----------
    model:
        MultiView模型

    train_loader:
        训练集

    val_loader:
        验证集

    epochs:
        默认30

    lr:
        CNN分类器学习率

    weight_decay:
        CNN分类器权重衰减

    device:
        cuda / cpu


    Returns
    -------
    best_state:
        Validation F1最佳模型参数

    best_mask:
        最佳模型对应VOC二值mask

    best_val_acc:
        最佳模型对应Validation Accuracy

    best_val_f1:
        最佳Validation F1
    """

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else
            "cpu"
        )

    model.to(device)


    # ============================================================
    # 1. Optimizer
    # ============================================================

    optimizer = torch.optim.Adam(
        [
            {
                "params":
                    model.logist.parameters(),

                "lr":
                    1e-5,

                "weight_decay":
                    1e-4
            },

            {
                "params":
                    model.classifier.parameters(),

                "lr":
                    lr,

                "weight_decay":
                    weight_decay
            }
        ]
    )


    # ============================================================
    # 2. 根据训练集类别数量计算class weight
    # ============================================================

    train_targets = []

    for _, y in train_loader:

        train_targets.append(
            y
        )

    train_targets = torch.cat(
        train_targets
    )


    class_counts = torch.bincount(
        train_targets,
        minlength=2
    ).float()


    class_weights = torch.sqrt(
        class_counts.sum()
        /
        (
            len(class_counts)
            *
            class_counts.clamp_min(1.0)
        )
    )


    class_weights = class_weights.to(
        device
    )


    print(
        "Train class counts:",
        class_counts.tolist()
    )

    print(
        "Class weights:",
        class_weights.detach().cpu().tolist()
    )


    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # ============================================================
    # 3. 最佳模型记录
    # ============================================================

    best_val_acc = 0.0

    best_val_f1 = -1.0

    best_threshold = 0.5

    best_state = None

    best_mask = None


    # ============================================================
    # 4. Epoch循环
    # ============================================================

    for epoch in range(epochs):


        # --------------------------------------------------------
        # temperature退火
        # 1.0 -> 0.3
        # --------------------------------------------------------

        if hasattr(
            model,
            "temperature"
        ):

            model.temperature = (
                1.0
                -
                0.7
                *
                epoch
                /
                max(
                    1,
                    epochs - 1
                )
            )


        # ========================================================
        # TRAIN
        # ========================================================

        model.train()


        train_loss_sum = 0.0

        train_sample_count = 0


        for x, y in train_loader:

            x = x.to(
                device
            )

            y = y.to(
                device
            ).long()


            optimizer.zero_grad()


            outputs, _ = model(
                x
            )


            loss = criterion(
                outputs,
                y
            )


            loss.backward()

            optimizer.step()


            train_loss_sum += (
                loss.item()
                *
                y.size(0)
            )

            train_sample_count += (
                y.size(0)
            )


        train_loss = (
            train_loss_sum
            /
            max(
                1,
                train_sample_count
            )
        )


        # ========================================================
        # VALIDATION
        # ========================================================

        model.eval()


        val_targets = []

        val_probs = []


        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(
                    device
                )

                y = y.to(
                    device
                ).long()


                outputs, _ = model(
                    x
                )


                probs = torch.softmax(
                    outputs,
                    dim=1
                )[:, 1]


                val_targets.extend(
                    y.detach()
                    .cpu()
                    .numpy()
                )


                val_probs.extend(
                    probs
                    .detach()
                    .cpu()
                    .numpy()
                )


        val_targets = np.asarray(
            val_targets
        )

        val_probs = np.asarray(
            val_probs
        )


        # ========================================================
        # 在 Validation 上搜索最佳分类阈值
        # ========================================================

        best_epoch_threshold = 0.5
        best_epoch_f1 = -1.0
        best_epoch_acc = 0.0


        for threshold in np.arange(
                0.20,
                0.81,
                0.01
        ):

            val_predictions = (
                val_probs >= threshold
            ).astype(int)

            current_f1 = float(
                f1_score(
                    val_targets,
                    val_predictions,
                    average="macro",
                    zero_division=0
                )
            )

            current_acc = float(
                np.mean(
                    val_targets
                    ==
                    val_predictions
                )
            )

            if (
                current_f1 > best_epoch_f1
                or
                (
                    np.isclose(
                        current_f1,
                        best_epoch_f1
                    )
                    and
                    current_acc > best_epoch_acc
                )
            ):

                best_epoch_f1 = current_f1
                best_epoch_acc = current_acc
                best_epoch_threshold = float(
                    threshold
                )


        val_f1 = best_epoch_f1
        val_acc = best_epoch_acc
        val_threshold = best_epoch_threshold


        


        # ========================================================
        # 5. 按Validation F1选择最佳模型
        #
        # F1相同时，再比较Accuracy
        # ========================================================

        is_better = (
            val_f1 > best_val_f1
            or
            (
                np.isclose(
                    val_f1,
                    best_val_f1
                )
                and
                val_acc > best_val_acc
            )
        )


        if is_better:

            best_val_f1 = val_f1

            best_val_acc = val_acc

            best_threshold = val_threshold


            best_state = copy.deepcopy(
                model.state_dict()
            )


            # ----------------------------------------------------
            # 保存当前最佳模型对应的稳定mask
            # ----------------------------------------------------

            with torch.no_grad():

                x_val = torch.cat(
                    [
                        x.to(
                            device
                        )
                        for x, _
                        in val_loader
                    ],
                    dim=0
                )


                model.eval()


                best_mask = (
                    model.get_score(
                        x_val
                    )
                    .detach()
                    .cpu()
                )


        # ========================================================
        # 打印
        # ========================================================

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss={train_loss:.4f} "
            f"Val Acc={val_acc:.4f} "
            f"Val Macro-F1={val_f1:.4f} "
            f"Threshold={val_threshold:.2f}"
        )


    # ============================================================
    # 6. 安全检查
    # ============================================================

    if best_state is None:

        raise RuntimeError(
            "训练结束后没有找到最佳模型。"
        )


    return (
        best_state,
        best_mask,
        best_val_acc,
        best_val_f1,
        best_threshold
    )