"""
数据划分模块

实现:
train / validation / test

比例:
train = 60%
val   = 20%
test  = 20%

采用分层划分，保证类别比例一致

"""

from typing import Sequence

import torch

from torch import Tensor

from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    ConcatDataset
)

from sklearn.model_selection import train_test_split



def stratified_split(
        X,
        y,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
):

    """
    分层三划分

    返回:
        train
        val
        test

    """

    assert (
        train_ratio
        +
        val_ratio
        +
        test_ratio
        ==
        1.0
    )


    indices = torch.arange(
        len(y)
    ).numpy()


    # 第一次:
    # train + val
    # test

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=y.numpy(),
        random_state=seed
    )


    # 第二次:
    # train
    # val

    val_ratio_adjust = (
        val_ratio
        /
        (train_ratio + val_ratio)
    )


    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adjust,
        stratify=y[train_val_idx].numpy(),
        random_state=seed
    )


    return (
        train_idx,
        val_idx,
        test_idx
    )



def split_dataset(
        samples: Tensor,
        labels: Tensor,
        batch_size=16,
        seed=42
):

    """
    返回:

    train_loader
    val_loader
    test_loader
    x_train


    x_train:
        只用于K-means聚类

    """


    train_idx, val_idx, test_idx = stratified_split(
        samples,
        labels,
        seed=seed
    )


    train_dataset = TensorDataset(
        samples[train_idx],
        labels[train_idx]
    )


    val_dataset = TensorDataset(
        samples[val_idx],
        labels[val_idx]
    )


    test_dataset = TensorDataset(
        samples[test_idx],
        labels[test_idx]
    )



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


    x_train = samples[train_idx]


    return (
        train_loader,
        val_loader,
        test_loader,
        x_train
    )