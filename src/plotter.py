"""
实验结果绘图模块

输入:
    test_predictions.npy
    test_metrics_per_repeat.csv

输出:
    ROC
    Confusion Matrix
    Metric boxplot

"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix
)



# =====================================================
# ROC曲线
# =====================================================

def plot_roc_curve(
        prediction_file,
        save_path
):

    data = np.load(
        prediction_file,
        allow_pickle=True
    )


    mean_fpr = np.linspace(
        0,
        1,
        200
    )


    tprs=[]
    aucs=[]


    for item in data:


        y_true=item["Targets"]

        probs=item["Probabilities"]


        fpr,tpr,_=roc_curve(
            y_true,
            probs
        )


        aucs.append(
            auc(fpr,tpr)
        )


        interp_tpr=np.interp(
            mean_fpr,
            fpr,
            tpr
        )

        interp_tpr[0]=0

        tprs.append(
            interp_tpr
        )


    mean_tpr=np.mean(
        tprs,
        axis=0
    )


    mean_auc=np.mean(
        aucs
    )


    plt.figure(
        figsize=(7,7)
    )


    plt.plot(
        mean_fpr,
        mean_tpr,
        label=f"AUC={mean_auc:.3f}"
    )


    plt.plot(
        [0,1],
        [0,1],
        "k--"
    )


    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )


    plt.title(
        "ROC Curve"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300
    )


    plt.close()



# =====================================================
# 混淆矩阵
# =====================================================

def plot_confusion_matrix(
        prediction_file,
        save_path
):


    data=np.load(
        prediction_file,
        allow_pickle=True
    )


    cms=[]


    for item in data:

        cm=confusion_matrix(

            item["Targets"],

            item["Predictions"],

            labels=[0,1]

        )

        cms.append(cm)



    mean_cm=np.mean(
        cms,
        axis=0
    )


    plt.figure(
        figsize=(5,5)
    )


    plt.imshow(
        mean_cm
    )


    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                f"{mean_cm[i,j]:.1f}",
                ha="center",
                va="center"
            )


    plt.xticks(
        [0,1],
        ["Pred 0","Pred 1"]
    )

    plt.yticks(
        [0,1],
        ["True 0","True 1"]
    )


    plt.title(
        "Average Confusion Matrix"
    )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300
    )


    plt.close()



# =====================================================
# 指标箱线图
# =====================================================

def plot_metrics_boxplot(
        csv_file,
        save_path
):


    df=pd.read_csv(
        csv_file
    )


    cols=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC"
    ]


    plt.figure(
        figsize=(8,5)
    )


    df[cols].boxplot()


    plt.ylabel(
        "Score"
    )


    plt.title(
        "Metrics Distribution"
    )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300
    )


    plt.close()