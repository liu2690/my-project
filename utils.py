import copy
from typing import Sequence
import pandas as pd
import numpy as np
import torch
import scipy.io as sio
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from modules import *
from torch.utils.data import random_split, ConcatDataset, DataLoader, TensorDataset, Dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from torch import Tensor


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def split_data(
    x: Tensor, y: Tensor, num_classes: int, split_length: Sequence[int | float]
) -> tuple[Dataset[tuple[Tensor, Tensor]],...]:
    datas = [random_split(TensorDataset(x[y == i], y[y == i]), split_length) for i in range(num_classes)]
    return tuple(ConcatDataset([d[i] for d in datas]) for i in range(len(split_length)))
# def split_dataset(samples: Tensor, labels: Tensor, batch_size = 16):
#     """
#     执行 6:2:2 的分层划分并返回 DataLoader
#     """
#     train_data, valid_data, test_data = split_data(samples, labels, 2, [0.6, 0.2, 0.2])
#     x_train = next(iter(DataLoader(train_data, batch_size=len(train_data), shuffle=False)))[0]
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
#     return train_loader, val_loader, test_loader, x_train #train_x用于k_means

def split_dataset(samples: Tensor, labels: Tensor, batch_size = None, split_length = [0.8, 0.2]):
    """
    执行 8:2 的分层划分并返回 DataLoader
    """
    train_data, test_data = split_data(samples, labels, 2, split_length)
    x_train = next(iter(DataLoader(train_data, batch_size=len(train_data), shuffle=False)))[0]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, x_train #train_x用于k_means


def feature_cluster(x, num_cluster = None):
    kmeans = KMeans(n_clusters=num_cluster, n_init=10)
    k_mean_mask = F.one_hot(torch.tensor(kmeans.fit_predict(x.T),dtype=torch.long))
    return k_mean_mask.T



def print_classification_result(cm: torch.Tensor) -> None:
    acc = (cm[:, 0, 0] + cm[:, 1, 1]) / cm.sum(dim=(1, 2))
    precision = cm[:, 1, 1] / (cm[:, 1, 1] + cm[:, 0, 1])
    recall = cm[:, 1, 1] / (cm[:, 1, 1] + cm[:, 1, 0])
    fpr = cm[:, 0, 1] / (cm[:, 0, 1] + cm[:, 0, 0])
    f1 = 2 * precision * recall / (precision + recall)
    n = len(cm)
    acc_std, acc_mean = torch.std_mean(acc)
    precision_std, precision_mean = torch.std_mean(precision)
    recall_std, recall_mean = torch.std_mean(recall)
    fpr_std, fpr_mean = torch.std_mean(fpr)
    f1_std, f1_mean = torch.std_mean(f1)
    sqrt_n = torch.sqrt(torch.tensor(n, dtype=float))
    print(f"acc: {acc_mean * 100:.2f}%±{1.96 * acc_std * 100 / sqrt_n:.2f}%.")
    print(f"precision: {precision_mean * 100:.2f}%±{1.96 * precision_std * 100 / sqrt_n:.2f}%.")
    print(f"recall: {recall_mean * 100:.2f}%±{1.96 * recall_std * 100 / sqrt_n:.2f}%.")
    print(f"fpr: {fpr_mean * 100:.2f}%±{1.96 * fpr_std * 100 / sqrt_n:.2f}%.")
    print(f"f1: {f1_mean * 100:.2f}%±{1.96 * f1_std * 100 / sqrt_n:.2f}%.")


def run_epoch(loader, classifier, optimizer=None):
    y_all = []
    y_softmax = []
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_hat = classifier(x)
        loss = F.cross_entropy(y_hat, y)
        if optimizer != None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        y_all.append(y)
        y_softmax.append(y_hat.detach())
        total_loss += loss.item() * len(x)
        total_count += len(x)
    return torch.concatenate(y_all), torch.concatenate(y_softmax), total_loss / total_count


def check_features(samples, labels, selector, epochs=1000, repeat=10):
    cm = []
    y_all = []
    y_softmax = []
    samples = samples[:, selector] if selector != None else samples
    for r in range(repeat):
        train_loader,  test_loader, x_train = split_dataset(samples, labels)
        classifier = nn.Sequential(
            nn.Linear(x_train.shape[1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
        ).to(device=DEVICE, dtype=samples.dtype)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=2.0e-3, weight_decay=2.0e-3, betas=(0.5, 0.7))
        best_model = None
        best_acc = 0.0
        for epoch in tqdm(range(epochs), desc=f"Repeat [{r+1}/{repeat}]"):
            _, _, train_loss = run_epoch(train_loader, classifier, optimizer)
            if (epoch + 1) % 100 == 0:
                run_epoch(test_loader, classifier, optimizer)
            with torch.no_grad():
                y_valid, y_hat_valid, _ = run_epoch(val_loader, classifier)
                
                acc = (y_valid == y_hat_valid.argmax(dim=1)).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(classifier.state_dict())
            # if (epoch + 1) % 100 == 0:
            #     print(f"Repeat [{r+1}/{repeat}] Epoch [{epoch+1}/{epochs}] | Loss: {train_loss:.4f} | Val Acc: {acc*100:.2f}% | Best Val: {best_acc*100:.2f}%")
        print(f"Repeat [{r+1}/{repeat}] | Best Val: {best_acc*100:.2f}%")
        assert best_model != None
        classifier.load_state_dict(best_model)
            
        with torch.no_grad():
            y_test, y_hat_test, _ = run_epoch(test_loader, classifier)
            cm.append(torch.tensor(confusion_matrix(y_test.cpu().numpy(), y_hat_test.argmax(dim=1).cpu().numpy())))
            y_all.append(y_test)
            y_softmax.append(y_hat_test)
    cm = torch.stack(cm)
    y_all = torch.stack(y_all)
    y_softmax = torch.stack(y_softmax)
    print_classification_result(cm)
    print()
    return cm, y_all, y_softmax
