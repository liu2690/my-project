#!/usr/bin/env python3
"""
对比不同分类器的实验结果

用法:
  python compare_results.py

读取各实验目录下的 metrics_summary.csv，生成对比表。
"""

import pandas as pd
import os
import sys


def main():
    paths = {
        'Original_Linear': './result/legacy/20/1+2_vs_3/metrics_summary.csv',
        'Deep_MLP':       './result/legacy/deep_mlp/1+2_vs_3/metrics_summary.csv',
        'CNN_Seq':        './result/legacy/cnn_seq/1+2_vs_3/metrics_summary.csv',
        'CNN_Cluster':    './result/legacy/cnn_cluster/1+2_vs_3/metrics_summary.csv',
    }

    rows = []
    for name, p in paths.items():
        if os.path.exists(p):
            df = pd.read_csv(p)
            for metric in ['F1', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity']:
                row = df[df['Metric'] == metric]
                if len(row):
                    rows.append({
                        'Model': name,
                        'Metric': metric,
                        'Mean': row['Mean'].values[0],
                        'CI95_Lower': row['CI95_Lower'].values[0],
                        'CI95_Upper': row['CI95_Upper'].values[0],
                    })
        else:
            print(f"[跳过] 未找到文件: {p}")

    if not rows:
        print("未找到任何实验结果文件，请先运行实验。")
        sys.exit(0)

    result_df = pd.DataFrame(rows)

    # Pivot 表
    pivot = result_df.pivot(index='Metric', columns='Model', values='Mean')
    print("\n" + "=" * 70)
    print(" 分类器对比 — Mean 指标")
    print("=" * 70)
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    # 保存
    out_dir = './result'
    os.makedirs(out_dir, exist_ok=True)
    result_df.to_csv(os.path.join(out_dir, 'comparison.csv'), index=False)
    print(f"\n对比结果已保存 → {os.path.join(out_dir, 'comparison.csv')}")

    # 判断
    models_present = result_df['Model'].unique()
    f1_data = result_df[result_df['Metric'] == 'F1']

    print("\n" + "=" * 70)
    print(" 判断")
    print("=" * 70)

    if 'CNN_Seq' in models_present and 'Deep_MLP' in models_present:
        cnn_f1 = f1_data[f1_data['Model'] == 'CNN_Seq']['Mean'].values[0]
        deep_f1 = f1_data[f1_data['Model'] == 'Deep_MLP']['Mean'].values[0]
        cnn_ci_low = f1_data[f1_data['Model'] == 'CNN_Seq']['CI95_Lower'].values[0]
        deep_ci_high = f1_data[f1_data['Model'] == 'Deep_MLP']['CI95_Upper'].values[0]

        if cnn_ci_low > deep_ci_high:
            print("CNN_Seq > Deep_MLP（95%CI 不重叠）→ CNN 参数共享归纳偏置有效，推荐 CNN_Seq")
        elif deep_ci_high >= cnn_f1 >= deep_f1:
            print("CNN_Seq ≈ Deep_MLP → 提升来自参数增加，推荐 Deep_MLP（更简单）")
        else:
            print("CNN_Seq 未优于 Deep_MLP → 推荐 Deep_MLP")

    if 'Original_Linear' in models_present:
        orig_f1 = f1_data[f1_data['Model'] == 'Original_Linear']['Mean'].values[0]
        best_f1 = f1_data['Mean'].max()
        if best_f1 <= orig_f1 + 0.01:
            print("所有新分类器均未显著优于 Original_Linear → 分类器架构非瓶颈，问题在特征选择阶段")


if __name__ == "__main__":
    main()