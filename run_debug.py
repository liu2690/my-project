#!/usr/bin/env python3
"""快速调试: 冠军数=1, 内层循环=1, epochs=2, 输出到 ./result/debug/"""

from voc_experiment import experiment


def main():
    experiment(
        './data/voc_dataset_1+2_vs_3.mat',
        './result/debug/1+2_vs_3',
        num_cluster=3,
        experiment_repeats=1,
        repeats=1,
        epochs=2,
        sparsity_lambda=1e-2,
        temp_start=1.0,
        temp_end=0.3,
        seed=42,
        panel_threshold=0.5
    )


if __name__ == "__main__":
    main()