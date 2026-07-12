#!/usr/bin/env python3
"""
Phase 1 诊断脚本：验证"分类器是否是瓶颈"

执行顺序：
  Step 1: 运行原始 Linear 分类器 + record_train_f1=True → 获取 train/val F1 gap
  Step 2: 运行 Deep MLP 基线 → 对比 val F1

输出：诊断报告到 ./result/diagnosis/diagnosis_report.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy.voc_experiment import experiment
import pandas as pd
import os
import sys


def main():
    data_path = './data/voc_dataset_1+2_vs_3.mat'
    base_dir = './result/diagnosis'

    # ================================================================
    # Step 1: 原始 Linear 分类器，开启 train F1 记录
    # ================================================================
    print("\n" + "=" * 70)
    print(" Phase 1 Step 1: 原始 Linear 分类器（train F1 诊断）")
    print("=" * 70)
    experiment(
        data_path,
        os.path.join(base_dir, 'linear'),
        num_cluster=3,
        experiment_repeats=10,   # 快速验证
        repeats=5,
        epochs=300,
        sparsity_lambda=1e-2,
        temp_start=1.0,
        temp_end=0.3,
        seed=42,
        panel_threshold=0.5,
        classifier_type='linear',
        record_train_f1=True,
    )

    # ================================================================
    # Step 2: Deep MLP 基线
    # ================================================================
    print("\n" + "=" * 70)
    print(" Phase 1 Step 2: Deep MLP 基线")
    print("=" * 70)
    experiment(
        data_path,
        os.path.join(base_dir, 'deep_mlp'),
        num_cluster=3,
        experiment_repeats=10,   # 快速验证
        repeats=5,
        epochs=300,
        sparsity_lambda=1e-2,
        temp_start=1.0,
        temp_end=0.3,
        seed=42,
        panel_threshold=0.5,
        classifier_type='deep_mlp',
        record_train_f1=True,
    )

    # ================================================================
    # 生成诊断报告
    # ================================================================
    print("\n" + "=" * 70)
    print(" 诊断报告")
    print("=" * 70)

    report_rows = []

    # 读取 val 指标
    for name, subdir in [('Linear', 'linear'), ('Deep_MLP', 'deep_mlp')]:
        csv_path = os.path.join(base_dir, subdir, 'metrics_summary.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for metric in ['F1', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity']:
                row = df[df['Metric'] == metric]
                if len(row):
                    report_rows.append({
                        'Model': name,
                        'Metric': metric,
                        'Mean': row['Mean'].values[0],
                        'CI95_Lower': row['CI95_Lower'].values[0],
                        'CI95_Upper': row['CI95_Upper'].values[0],
                    })

    # 读取 train F1
    for name, subdir in [('Linear', 'linear'), ('Deep_MLP', 'deep_mlp')]:
        csv_path = os.path.join(base_dir, subdir, 'train_f1_diagnosis.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            report_rows.append({
                'Model': name,
                'Metric': 'Train_F1',
                'Mean': df['Train_F1'].mean(),
                'CI95_Lower': df['Train_F1'].mean() - df['Train_F1'].std(),
                'CI95_Upper': df['Train_F1'].mean() + df['Train_F1'].std(),
            })
            report_rows.append({
                'Model': name,
                'Metric': 'Train_Acc',
                'Mean': df['Train_Acc'].mean(),
                'CI95_Lower': df['Train_Acc'].mean() - df['Train_Acc'].std(),
                'CI95_Upper': df['Train_Acc'].mean() + df['Train_Acc'].std(),
            })

    report_df = pd.DataFrame(report_rows)
    report_csv = os.path.join(base_dir, 'diagnosis_report.csv')
    report_df.to_csv(report_csv, index=False)
    print(f"\n诊断报告已保存 → {report_csv}")

    # 打印 pivot 表
    pivot = report_df.pivot(index='Metric', columns='Model', values='Mean')
    print("\n" + pivot.to_string())

    # 判断
    linear_f1 = report_df[(report_df['Model'] == 'Linear') & (report_df['Metric'] == 'F1')]
    deepmlp_f1 = report_df[(report_df['Model'] == 'Deep_MLP') & (report_df['Metric'] == 'F1')]

    if len(linear_f1) and len(deepmlp_f1):
        linear_val_f1 = linear_f1['Mean'].values[0]
        deepmlp_val_f1 = deepmlp_f1['Mean'].values[0]
        delta = deepmlp_val_f1 - linear_val_f1

        print(f"\n=== BPD 分类器瓶颈诊断报告 ===")
        print(f"原始 Linear: val F1 = {linear_val_f1:.4f}")
        print(f"Deep MLP:    val F1 = {deepmlp_val_f1:.4f}")
        print(f"ΔF1 = {delta:+.4f}")

        if delta > 0.02:
            conclusion = "分类器是瓶颈 —— 建议进入 Phase 2，实现 CNN"
        elif delta > -0.02:
            conclusion = "分类器不是瓶颈 —— 放弃 CNN，优化 mask 质量或训练策略"
        else:
            conclusion = "Deep MLP 过拟合 —— 减小宽度或放弃 CNN"

        print(f"结论: {conclusion}")

        # 写入结论文件
        with open(os.path.join(base_dir, 'conclusion.txt'), 'w') as f:
            f.write(f"=== BPD 分类器瓶颈诊断报告 ===\n")
            f.write(f"原始 Linear: val F1 = {linear_val_f1:.4f}\n")
            f.write(f"Deep MLP:    val F1 = {deepmlp_val_f1:.4f}\n")
            f.write(f"ΔF1 = {delta:+.4f}\n")
            f.write(f"结论: {conclusion}\n")
        print(f"\n结论已保存 → {os.path.join(base_dir, 'conclusion.txt')}")


if __name__ == "__main__":
    main()