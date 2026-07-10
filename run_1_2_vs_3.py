#!/usr/bin/env python3
"""
正式实验: 1+2 vs 3

用法:
  python run_1_2_vs_3.py                    # 默认: 原始 Linear 分类器
  python run_1_2_vs_3.py --cnn_seq          # 路线 A: CNN Seq
  python run_1_2_vs_3.py --cnn_cluster      # 路线 B: CNN Cluster
  python run_1_2_vs_3.py --deep_mlp          # 路线 C: Deep MLP
  python run_1_2_vs_3.py --all               # 运行全部四种

输出目录:
  ./result/20/1+2_vs_3/          → Linear
  ./result/cnn_seq/1+2_vs_3/     → CNN Seq
  ./result/cnn_cluster/1+2_vs_3/ → CNN Cluster
  ./result/deep_mlp/1+2_vs_3/    → Deep MLP
"""

import sys
from voc_experiment import experiment

DATA_PATH = './data/voc_dataset_1+2_vs_3.mat'
NUM_CLUSTER = 3
EXPERIMENT_REPEATS = 100
REPEATS = 20
EPOCHS = 300
SPARSITY_LAMBDA = 1e-2
TEMP_START = 1.0
TEMP_END = 0.3
SEED = 42
PANEL_THRESHOLD = 0.5


def run_experiment(classifier_type, save_path):
    print(f"\n{'='*70}")
    print(f" 运行实验: classifier_type={classifier_type}")
    print(f" 输出目录: {save_path}")
    print(f"{'='*70}")
    experiment(
        DATA_PATH,
        save_path,
        num_cluster=NUM_CLUSTER,
        experiment_repeats=EXPERIMENT_REPEATS,
        repeats=REPEATS,
        epochs=EPOCHS,
        sparsity_lambda=SPARSITY_LAMBDA,
        temp_start=TEMP_START,
        temp_end=TEMP_END,
        seed=SEED,
        panel_threshold=PANEL_THRESHOLD,
        classifier_type=classifier_type,
        record_train_f1=False,
    )


def main():
    args = set(sys.argv[1:])

    if '--all' in args:
        # 运行全部四种
        run_experiment('linear',       './result/20/1+2_vs_3')
        run_experiment('deep_mlp',     './result/deep_mlp/1+2_vs_3')
        run_experiment('cnn_seq',      './result/cnn_seq/1+2_vs_3')
        run_experiment('cnn_cluster',  './result/cnn_cluster/1+2_vs_3')
    elif '--cnn_seq' in args:
        run_experiment('cnn_seq', './result/cnn_seq/1+2_vs_3')
    elif '--cnn_cluster' in args:
        run_experiment('cnn_cluster', './result/cnn_cluster/1+2_vs_3')
    elif '--deep_mlp' in args:
        run_experiment('deep_mlp', './result/deep_mlp/1+2_vs_3')
    else:
        # 默认: 原始 Linear
        run_experiment('linear', './result/20/1+2_vs_3')


if __name__ == "__main__":
    main()