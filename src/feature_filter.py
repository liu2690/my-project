"""
训练集特征预筛选

只使用 train 数据计算特征分数，
再把相同特征列应用到 train / val / test。
"""

import torch
from sklearn.feature_selection import SelectKBest, f_classif


def select_top_k_features(
        x_train,
        y_train,
        top_k=200
):
    """
    Parameters
    ----------
    x_train:
        Tensor [n_train, n_features]

    y_train:
        Tensor [n_train]

    top_k:
        保留的VOC数量

    Returns
    -------
    selected_indices:
        被选择VOC的列索引
    """

    n_features = x_train.shape[1]

    if top_k is None or top_k >= n_features:
        return torch.arange(
            n_features,
            dtype=torch.long
        )

    selector = SelectKBest(
        score_func=f_classif,
        k=top_k
    )

    selector.fit(
        x_train.cpu().numpy(),
        y_train.cpu().numpy()
    )

    selected_indices = selector.get_support(
        indices=True
    )

    return torch.tensor(
        selected_indices,
        dtype=torch.long
    )