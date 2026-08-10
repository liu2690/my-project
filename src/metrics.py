"""
评价指标模块

解决:
Precision/F1 NaN问题

"""


import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)



def compute_metrics(
        targets,
        predictions,
        probabilities
):

    """
    计算分类指标

    zero_division=0:
        防止Precision/F1出现nan

    """


    targets = np.asarray(
        targets
    )

    predictions = np.asarray(
        predictions
    )

    probabilities = np.asarray(
        probabilities
    )


    cm = confusion_matrix(
        targets,
        predictions,
        labels=[0,1]
    )


    tn, fp = cm[0]

    fn, tp = cm[1]


    sensitivity = (
        tp /
        (tp+fn)
        if tp+fn>0
        else 0
    )


    specificity = (
        tn /
        (tn+fp)
        if tn+fp>0
        else 0
    )


    precision = precision_score(
        targets,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        targets,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        targets,
        predictions,
        zero_division=0
    )


    accuracy = accuracy_score(
        targets,
        predictions
    )


    try:

        auc = roc_auc_score(
            targets,
            probabilities
        )

    except ValueError:

        auc = 0.0



    return {

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "Accuracy":
            accuracy,

        "AUC":
            auc,

        "Confusion_Matrix":
            cm

    }