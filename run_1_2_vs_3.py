#!/usr/bin/env python3
"""正式实验: 1+2 vs 3, 冠军数=100, 内层循环=20, 输出到 ./result/20/"""

from voc_experiment import experiment


def main():
    experiment(
        './data/voc_dataset_1+2_vs_3.mat',
        './result/20/1+2_vs_3',
        num_cluster=3,
        experiment_repeats=100,
        repeats=20,
        epochs=300,
        sparsity_lambda=1e-2,
        temp_start=1.0,
        temp_end=0.3,
        seed=42,
        panel_threshold=0.5
    )


if __name__ == "__main__":
    main()