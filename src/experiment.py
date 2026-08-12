"""
多次重复实验入口

当前流程：

1. 自动选择是否使用 Unknown 数据
2. 固定独立 Test = 20%
3. 剩余 80% 使用 Stratified K-Fold
4. 每个 Fold 内：
   - 只用 Fold Train 做 Top-K VOC 筛选
   - 只用 Fold Train 做 K-means
   - 训练 MultiView + CNN
   - Validation 上选择最佳模型与分类阈值
5. 选择 CV 中表现最好的 Fold
6. 在固定 Test 上做最终评价
7. 多个 Repeat 汇总结果
"""

import os
import json
import random

import numpy as np
import pandas as pd

import torch
import scipy.io as sio

from src.split import (
    split_trainval_test,
    make_cv_folds,
    make_loader,
)

from src.clustering import feature_cluster
from src.model import MultiView
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.feature_filter import select_top_k_features


# ============================================================
# 随机种子
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# 根据特征索引裁剪 Tensor
# ============================================================

def apply_indices_to_tensor(
        X,
        selected_indices
):

    return X[:, selected_indices]


# ============================================================
# 单个 Repeat：
# 固定 Test + K-Fold CV
# ============================================================

def run_cv_experiment(

        X,
        y,

        repeat_seed,

        num_cluster=8,

        top_k_features=100,

        epochs=100,

        batch_size=16,

        cv_folds=5
):

    set_seed(
        repeat_seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    # ========================================================
    # 1. 固定独立 Test
    # ========================================================

    (
        X_trainval,
        y_trainval,
        X_test,
        y_test,

    ) = split_trainval_test(

        X,
        y,

        test_ratio=0.2,

        seed=repeat_seed
    )

    print(
        f"TrainVal: {len(X_trainval)}, "
        f"Test: {len(X_test)}"
    )

    # ========================================================
    # 2. 记录 CV 最佳模型
    # ========================================================

    best_cv_f1 = -1.0

    best_cv_acc = 0.0

    best_checkpoint = None

    best_fold = None

    # ========================================================
    # 3. Stratified K-Fold
    # ========================================================

    for fold_idx, (

        X_train,
        y_train,
        X_val,
        y_val,

    ) in enumerate(

        make_cv_folds(

            X_trainval,
            y_trainval,

            n_splits=cv_folds,

            seed=repeat_seed
        ),

        start=1
    ):

        print(
            "\n"
            + "-" * 50
        )

        print(
            f"CV Fold {fold_idx}/{cv_folds}"
        )

        print(
            f"Train={len(X_train)}, "
            f"Val={len(X_val)}"
        )

        # ====================================================
        # 4. Top-K
        #
        # 注意：
        # 这里只能使用当前 fold 的训练数据
        # ====================================================

        selected_indices = select_top_k_features(

            X_train,
            y_train,

            top_k=top_k_features
        )

        X_train_selected = apply_indices_to_tensor(

            X_train,
            selected_indices
        )

        X_val_selected = apply_indices_to_tensor(

            X_val,
            selected_indices
        )

        print(
            f"Top-K features: "
            f"{X.shape[1]} -> "
            f"{X_train_selected.shape[1]}"
        )

        # ====================================================
        # 5. K-means
        #
        # 同样只能使用当前 fold 的训练数据
        # ====================================================

        cluster_mask = feature_cluster(

            X_train_selected
            .detach()
            .cpu()
            .numpy(),

            num_cluster=num_cluster,

            seed=(
                repeat_seed
                +
                fold_idx
            )
        )

        print(
            "Cluster mask shape:",
            cluster_mask.shape
        )

        # ====================================================
        # 6. DataLoader
        # ====================================================

        train_loader = make_loader(

            X_train_selected,
            y_train,

            batch_size=batch_size,

            shuffle=True
        )

        val_loader = make_loader(

            X_val_selected,
            y_val,

            batch_size=batch_size,

            shuffle=False
        )

        # ====================================================
        # 7. 创建 MultiView + CNN
        # ====================================================

        model = MultiView(

            cluster_mask=cluster_mask,

            num_blocks=8,

            head_dim=128,

            num_classes=2,

            temperature=1.0
        )

        # ====================================================
        # 8. Fold 内训练
        # ====================================================

        (
            best_state,
            best_mask,
            best_val_acc,
            best_val_f1,
            best_threshold,

        ) = train_model(

            model,

            train_loader,

            val_loader,

            epochs=epochs,

            device=device
        )

        print(
            f"Fold {fold_idx} Best: "
            f"Acc={best_val_acc:.4f}, "
            f"Macro-F1={best_val_f1:.4f}, "
            f"Threshold={best_threshold:.2f}"
        )

        # ====================================================
        # 9. 选择整个 CV 中最佳 Fold
        # ====================================================

        is_better = (

            best_val_f1 > best_cv_f1

            or

            (
                np.isclose(
                    best_val_f1,
                    best_cv_f1
                )

                and

                best_val_acc > best_cv_acc
            )
        )

        if is_better:

            best_cv_f1 = (
                best_val_f1
            )

            best_cv_acc = (
                best_val_acc
            )

            best_fold = (
                fold_idx
            )

            selected_indices_cpu = (
                selected_indices
                .detach()
                .cpu()
            )

            best_checkpoint = {

                "state_dict":
                    best_state,

                "mask":
                    best_mask,

                "cluster_mask":
                    cluster_mask,

                "selected_indices":
                    selected_indices_cpu,

                "val_acc":
                    best_val_acc,

                "val_f1":
                    best_val_f1,

                "threshold":
                    best_threshold,

                "fold":
                    fold_idx,

                "repeat_seed":
                    repeat_seed
            }

    # ========================================================
    # 10. 检查是否得到有效模型
    # ========================================================

    if best_checkpoint is None:

        raise RuntimeError(
            "K-Fold CV 没有产生有效模型。"
        )

    print(
        "\n"
        + "=" * 50
    )

    print(
        f"Best CV Fold: "
        f"{best_fold}"
    )

    print(
        f"Best CV Accuracy: "
        f"{best_cv_acc:.4f}"
    )

    print(
        f"Best CV Macro-F1: "
        f"{best_cv_f1:.4f}"
    )

    print(
        f"Best Threshold: "
        f"{best_checkpoint['threshold']:.2f}"
    )

    # ========================================================
    # 11. Test 使用最佳 Fold 的 Top-K
    # ========================================================

    selected_indices = (
        best_checkpoint[
            "selected_indices"
        ]
    )

    X_test_selected = apply_indices_to_tensor(

        X_test,
        selected_indices
    )

    test_loader = make_loader(

        X_test_selected,
        y_test,

        batch_size=batch_size,

        shuffle=False
    )

    # ========================================================
    # 12. 重建最佳 Fold 模型
    # ========================================================

    model = MultiView(

        cluster_mask=
            best_checkpoint[
                "cluster_mask"
            ],

        num_blocks=8,

        head_dim=128,

        num_classes=2,

        temperature=1.0
    )

    model.load_state_dict(

        best_checkpoint[
            "state_dict"
        ]
    )

    # ========================================================
    # 13. 固定 Test 最终评价
    # ========================================================

    test_result = evaluate_model(

        model,

        test_loader,

        threshold=
            best_checkpoint[
                "threshold"
            ],

        device=device
    )

    test_result[
        "CV_Accuracy"
    ] = float(
        best_cv_acc
    )

    test_result[
        "CV_F1"
    ] = float(
        best_cv_f1
    )

    test_result[
        "CV_Best_Fold"
    ] = int(
        best_fold
    )

    test_result[
        "Val_Threshold"
    ] = float(
        best_checkpoint[
            "threshold"
        ]
    )

    test_result[
        "Seed"
    ] = int(
        repeat_seed
    )

    return (
        test_result,
        best_checkpoint
    )


# ============================================================
# 总实验
# ============================================================

def experiment(

        data_path=None,

        save_path="result/default",

        use_unknown=True,

        num_cluster=8,

        top_k_features=100,

        epochs=100,

        batch_size=16,

        cv_folds=5,

        experiment_repeats=5,

        seed=42
):

    os.makedirs(
        save_path,
        exist_ok=True
    )

    model_dir = os.path.join(

        save_path,
        "models"
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    # ========================================================
    # 1. 自动选择数据
    # ========================================================

    if data_path is None:

        if use_unknown:

            data_path = (
                "data/"
                "voc_dataset_1+2_vs_3_with_unknown.mat"
            )

        else:

            data_path = (
                "data/"
                "voc_dataset_1+2_vs_3.mat"
            )

    # ========================================================
    # 2. 加载数据
    # ========================================================

    data = sio.loadmat(
        data_path
    )

    X = torch.tensor(

        data["X"],

        dtype=torch.float32
    )

    y = torch.tensor(

        data["y"].reshape(-1),

        dtype=torch.long
    )

    print(
        "=" * 60
    )

    print(
        "Experiment start"
    )

    print(
        "Samples:",
        X.shape[0]
    )

    print(
        "Features:",
        X.shape[1]
    )

    print(
        "Unknown:",
        use_unknown
    )

    print(
        "Top-K:",
        top_k_features
    )

    print(
        "CV folds:",
        cv_folds
    )

    print(
        "Repeats:",
        experiment_repeats
    )

    # ========================================================
    # 3. 结果容器
    # ========================================================

    all_results = []

    all_predictions = []

    all_masks = []

    # ========================================================
    # 4. Repeat
    # ========================================================

    for repeat in range(
        experiment_repeats
    ):

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Repeat "
            f"{repeat + 1}/"
            f"{experiment_repeats}"
        )

        (
            result,
            checkpoint,

        ) = run_cv_experiment(

            X,
            y,

            repeat_seed=
                seed + repeat,

            num_cluster=
                num_cluster,

            top_k_features=
                top_k_features,

            epochs=
                epochs,

            batch_size=
                batch_size,

            cv_folds=
                cv_folds
        )

        all_results.append(
            result
        )

        # ====================================================
        # 保存预测结果
        # ====================================================

        all_predictions.append(

            {
                "Repeat":
                    repeat,

                "Targets":
                    result[
                        "Targets"
                    ],

                "Predictions":
                    result[
                        "Predictions"
                    ],

                "Probabilities":
                    result[
                        "Probabilities"
                    ],

                "Threshold":
                    result[
                        "Threshold"
                    ]
            }
        )

        # ====================================================
        # 将 Top-K mask 映射回原始特征维度
        # ====================================================

        full_mask = np.zeros(

            X.shape[1],

            dtype=np.float32
        )

        selected_indices_np = (

            checkpoint[
                "selected_indices"
            ]
            .cpu()
            .numpy()
        )

        local_mask = (

            checkpoint[
                "mask"
            ]
            .cpu()
            .numpy()
        )

        full_mask[
            selected_indices_np
        ] = local_mask

        all_masks.append(
            full_mask
        )

        # ====================================================
        # 保存最佳模型
        # ====================================================

        torch.save(

            checkpoint,

            os.path.join(

                model_dir,

                f"best_model_repeat"
                f"{repeat}.pt"
            )
        )

        print(
            "\nTest result:"
        )

        print(
            result
        )

    # ========================================================
    # 5. 保存逐 Repeat 指标
    # ========================================================

    df = pd.DataFrame(
        all_results
    )

    df.to_csv(

        os.path.join(

            save_path,

            "test_metrics_per_repeat.csv"
        ),

        index=False
    )

    # ========================================================
    # 6. 保存原始预测
    # ========================================================

    np.save(

        os.path.join(

            save_path,

            "test_predictions.npy"
        ),

        np.array(
            all_predictions,
            dtype=object
        ),

        allow_pickle=True
    )

    # ========================================================
    # 7. 保存特征 mask
    # ========================================================

    np.save(

        os.path.join(

            save_path,

            "selection_masks.npy"
        ),

        np.array(
            all_masks
        )
    )

    # ========================================================
    # 8. Test 指标汇总
    # ========================================================

    metrics = [

        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC",
        "Sensitivity",
        "Specificity"
    ]

    summary = []

    for metric in metrics:

        values = np.asarray(

            df[
                metric
            ].values,

            dtype=float
        )

        mean = float(
            np.mean(
                values
            )
        )

        # 只有一个 repeat 时避免 ddof=1 警告
        if len(values) >= 2:

            std = float(
                np.std(
                    values,
                    ddof=1
                )
            )

            ci = float(
                1.96
                *
                std
                /
                np.sqrt(
                    len(values)
                )
            )

        else:

            std = 0.0

            ci = 0.0

        summary.append(

            {
                "Metric":
                    metric,

                "Mean":
                    mean,

                "Std":
                    std,

                "CI95_Lower":
                    max(
                        0.0,
                        mean - ci
                    ),

                "CI95_Upper":
                    min(
                        1.0,
                        mean + ci
                    )
            }
        )

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(

        os.path.join(

            save_path,

            "test_metrics_summary.csv"
        ),

        index=False
    )

    # ========================================================
    # 9. 保存实验配置
    # ========================================================

    config = {

        "data_path":
            data_path,

        "use_unknown":
            use_unknown,

        "num_cluster":
            num_cluster,

        "top_k_features":
            top_k_features,

        "epochs":
            epochs,

        "batch_size":
            batch_size,

        "cv_folds":
            cv_folds,

        "experiment_repeats":
            experiment_repeats,

        "seed":
            seed,

        "test_ratio":
            0.2
    }

    with open(

        os.path.join(

            save_path,

            "config.json"
        ),

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            config,

            f,

            indent=2,

            ensure_ascii=False
        )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Finished"
    )

    print(
        summary_df
    )

    return summary_df