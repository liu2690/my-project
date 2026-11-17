
import numpy as np
import pandas as pd

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# 读取数据
data = loadmat(
    './data/voc_dataset_table1_1+2_vs_3_bpd.mat'
)

X = data['X']
y = data['y'].reshape(-1)

# 读取264个VOC
mask = np.load(
    './result/table1_10x10/feature_panels/mask_gt60.npy'
).astype(bool)

X = X[:, mask]

# 固定开发集和内部训练验证划分
split = np.load(
    './result/table1_10x10/fixed_outer_split.npz'
)

development_indices = split['development_indices']

train_indices, val_indices = train_test_split(
    development_indices,
    test_size=0.25,
    random_state=1042,
    stratify=y[development_indices]
)

X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[val_indices]
y_val = y[val_indices]

# 不同L2正则化强度
results = []

for C in [0.001, 0.01, 0.1, 1, 10]:

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty='l2',
            C=C,
            solver='liblinear',
            max_iter=5000,
            random_state=42
        )
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    results.append({
        'C': C,
        'Accuracy': accuracy_score(y_val, predictions),
        'Sensitivity': tp / max(tp + fn, 1),
        'Specificity': tn / max(tn + fp, 1),
        'F1': f1_score(y_val, predictions, zero_division=0),
        'AUC': roc_auc_score(y_val, probabilities)
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

results_df.to_csv(
    './result/logistic_regression_264.csv',
    index=False
)
