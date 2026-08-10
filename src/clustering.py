"""
VOC 特征聚类模块

使用 K-means 对VOC特征进行分组

输入:
    X:
        samples × features

输出:
    cluster mask:
        clusters × features

"""

import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans



def feature_cluster(
        x,
        num_cluster=8,
        seed=42
):

    """
    K-means VOC聚类

    Parameters
    ----------
    x:
        numpy array
        shape:
        [samples, features]

    num_cluster:
        K值

    """

    kmeans = KMeans(
        n_clusters=num_cluster,
        n_init=10,
        random_state=seed
    )


    # 对VOC特征聚类
    labels = kmeans.fit_predict(
        x.T
    )


    cluster_mask = F.one_hot(
        torch.tensor(
            labels,
            dtype=torch.long
        ),
        num_classes=num_cluster
    )


    return cluster_mask.T