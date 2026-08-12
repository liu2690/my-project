"""
数据划分模块

支持两种实验方式：

1. 原来的 train / val / test 三划分
   train = 60%
   val   = 20%
   test  = 20%

2. 固定独立 Test + Stratified K-Fold
   test = 20%
   剩余 80% 用于 K-Fold 训练 / 验证

所有划分均采用分层方式，
尽量保持各数据子集中的类别比例一致。
"""

import numpy as np
import torch

from torch import Tensor

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)


# ============================================================
# 1. 固定独立 Test
# ============================================================

def split_trainval_test(
        samples: Tensor,
        labels: Tensor,
        test_ratio=0.2,
        seed=42
):
    """
    先从全部数据中固定划分独立 Test。

    Parameters
    ----------
    samples:
        特征矩阵
        shape = [n_samples, n_features]

    labels:
        标签
        shape = [n_samples]

    test_ratio:
        Test 比例
        默认 0.2

    seed:
        随机种子

    Returns
    -------
    X_trainval:
        剩余 train + validation 数据

    y_trainval:
        train + validation 标签

    X_test:
        独立 Test 数据

    y_test:
        独立 Test 标签
    """

    indices = np.arange(
        len(labels)
    )

    labels_np = (
        labels
        .detach()
        .cpu()
        .numpy()
    )

    trainval_idx, test_idx = train_test_split(
        indices,

        test_size=test_ratio,

        stratify=labels_np,

        random_state=seed
    )

    X_trainval = samples[
        trainval_idx
    ]

    y_trainval = labels[
        trainval_idx
    ]

    X_test = samples[
        test_idx
    ]

    y_test = labels[
        test_idx
    ]

    return (
        X_trainval,
        y_trainval,
        X_test,
        y_test
    )


# ============================================================
# 2. Stratified K-Fold
# ============================================================

def make_cv_folds(
        X_trainval: Tensor,
        y_trainval: Tensor,
        n_splits=5,
        seed=42
):
    """
    对 train_val_pool 做 Stratified K-Fold。

    每一折返回：

        X_train
        y_train
        X_val
        y_val

    Parameters
    ----------
    X_trainval:
        固定 Test 后剩余的数据

    y_trainval:
        对应标签

    n_splits:
        K-Fold 折数
        默认 5

    seed:
        随机种子
    """

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    X_np = (
        X_trainval
        .detach()
        .cpu()
        .numpy()
    )

    y_np = (
        y_trainval
        .detach()
        .cpu()
        .numpy()
    )

    for train_idx, val_idx in skf.split(
        X_np,
        y_np
    ):

        X_train = X_trainval[
            train_idx
        ]

        y_train = y_trainval[
            train_idx
        ]

        X_val = X_trainval[
            val_idx
        ]

        y_val = y_trainval[
            val_idx
        ]

        yield (
            X_train,
            y_train,
            X_val,
            y_val
        )


# ============================================================
# 3. Tensor -> DataLoader
# ============================================================

def make_loader(
        X: Tensor,
        y: Tensor,
        batch_size=16,
        shuffle=False
):
    """
    根据 Tensor 创建 DataLoader。
    """

    dataset = TensorDataset(
        X,
        y
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader


# ============================================================
# 4. 原来的 60 / 20 / 20 分层划分
# ============================================================

def stratified_split(
        X: Tensor,
        y: Tensor,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
):
    """
    原来的 train / val / test 分层三划分。

    默认比例：

        train = 60%
        val   = 20%
        test  = 20%

    Returns
    -------
    train_idx
    val_idx
    test_idx
    """

    total_ratio = (
        train_ratio
        +
        val_ratio
        +
        test_ratio
    )

    if not np.isclose(
        total_ratio,
        1.0
    ):
        raise ValueError(
            "train_ratio + val_ratio + "
            "test_ratio 必须等于 1.0"
        )

    indices = np.arange(
        len(y)
    )

    y_np = (
        y
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # 第一次：
    # train + val
    # vs
    # test
    # --------------------------------------------------------

    train_val_idx, test_idx = train_test_split(
        indices,

        test_size=test_ratio,

        stratify=y_np,

        random_state=seed
    )

    # --------------------------------------------------------
    # 第二次：
    # train
    # vs
    # val
    # --------------------------------------------------------

    val_ratio_adjusted = (
        val_ratio
        /
        (
            train_ratio
            +
            val_ratio
        )
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,

        test_size=
            val_ratio_adjusted,

        stratify=
            y_np[
                train_val_idx
            ],

        random_state=seed
    )

    return (
        train_idx,
        val_idx,
        test_idx
    )


# ============================================================
# 5. 原来的 split_dataset
# ============================================================

def split_dataset(
        samples: Tensor,
        labels: Tensor,
        batch_size=16,
        seed=42
):
    """
    原来的 60 / 20 / 20 数据划分接口。

    返回：

        train_loader
        val_loader
        test_loader
        x_train
        y_train

    其中：

        x_train
        y_train

    可以用于：

        Top-K 特征筛选
        K-means
    """

    (
        train_idx,
        val_idx,
        test_idx

    ) = stratified_split(
        samples,
        labels,
        seed=seed
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        samples[
            train_idx
        ],
        labels[
            train_idx
        ]
    )

    val_dataset = TensorDataset(
        samples[
            val_idx
        ],
        labels[
            val_idx
        ]
    )

    test_dataset = TensorDataset(
        samples[
            test_idx
        ],
        labels[
            test_idx
        ]
    )

    # --------------------------------------------------------
    # Loader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # --------------------------------------------------------
    # Train Tensor
    # --------------------------------------------------------

    x_train = samples[
        train_idx
    ]

    y_train = labels[
        train_idx
    ]

    return (
        train_loader,
        val_loader,
        test_loader,
        x_train,
        y_train
    )


# ============================================================
# 6. 简单测试
# ============================================================

if __name__ == "__main__":

    X = torch.randn(
        159,
        780
    )

    y = torch.tensor(
        [0] * 105
        +
        [1] * 54,
        dtype=torch.long
    )

    print(
        "=" * 60
    )

    print(
        "Test: old split_dataset"
    )

    (
        train_loader,
        val_loader,
        test_loader,
        x_train,
        y_train

    ) = split_dataset(
        X,
        y,
        batch_size=16,
        seed=42
    )

    print(
        "Train:",
        len(
            train_loader.dataset
        )
    )

    print(
        "Val:",
        len(
            val_loader.dataset
        )
    )

    print(
        "Test:",
        len(
            test_loader.dataset
        )
    )

    print(
        "x_train:",
        x_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )


    print(
        "\n"
        +
        "=" * 60
    )

    print(
        "Test: fixed test + 5-fold CV"
    )

    (
        X_trainval,
        y_trainval,
        X_test,
        y_test

    ) = split_trainval_test(
        X,
        y,
        test_ratio=0.2,
        seed=42
    )

    print(
        "TrainVal:",
        X_trainval.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    for fold_idx, (
        X_train_fold,
        y_train_fold,
        X_val_fold,
        y_val_fold

    ) in enumerate(
        make_cv_folds(
            X_trainval,
            y_trainval,
            n_splits=5,
            seed=42
        ),
        start=1
    ):

        print(
            f"Fold {fold_idx}: "
            f"Train={len(X_train_fold)}, "
            f"Val={len(X_val_fold)}"
        )