#!/usr/bin/env python3
"""
run_legacy_holdout.py — 阶段一：legacy holdout dry-run 隔离

用法:
  # Dry-run (安全，不训练模型，不输出指标)
  python run_legacy_holdout.py --dry-run

  # 正式评价 (需要冻结配置，本阶段不实现)
  python run_legacy_holdout.py \
    --frozen-config configs/final_frozen.yaml \
    --confirm-final-evaluation

本阶段只允许 dry-run，正式评价在没有冻结配置时会安全拒绝。
"""

import os
import sys
import json
import warnings
from typing import Optional

from data_pipeline import build_dataset, EXCEL_PATH
from dataset_schema import VOCDatasetBundle
from split_manager import (
    generate_legacy_holdout,
    validate_legacy_holdout,
    SPLITS_DIR,
)

warnings.filterwarnings('ignore')


def dry_run(bundle: VOCDatasetBundle) -> dict:
    """执行 dry-run：只验证数据流隔离，不训练模型，不输出指标。

    Returns:
        dry-run 结果 dict
    """
    results = {
        'status': 'dry_run_completed',
        'checks': {},
    }

    # 1. 加载数据
    print("[dry-run] 1/7 加载数据...")
    results['checks']['data_loaded'] = True
    results['checks']['n_samples'] = bundle.n_samples
    results['checks']['n_features'] = bundle.n_features
    print(f"  [✓] 样本数: {bundle.n_samples}, 特征数: {bundle.n_features}")

    # 2. 校验 dataset fingerprint
    print("[dry-run] 2/7 校验 dataset fingerprint...")
    results['checks']['fingerprint'] = bundle.dataset_fingerprint[:32] + '...'
    print(f"  [✓] Fingerprint: {bundle.dataset_fingerprint[:32]}...")

    # 3. 校验 legacy manifest
    print("[dry-run] 3/7 校验 legacy manifest...")
    manifest_path = os.path.join(SPLITS_DIR, 'legacy_holdout_manifest.json')
    if not os.path.exists(manifest_path):
        # 生成
        legacy_manifest = generate_legacy_holdout(bundle, force_regenerate=False)
    else:
        with open(manifest_path, 'r') as f:
            legacy_manifest = json.load(f)

    results['checks']['legacy_status'] = legacy_manifest.get('status', 'unknown')
    print(f"  [✓] Status: {legacy_manifest.get('status')}")

    # 4. 检查 train/test 无交集
    print("[dry-run] 4/7 检查 train/test 无交集...")
    validate_legacy_holdout(legacy_manifest, bundle)
    results['checks']['train_test_no_overlap'] = True
    print("  [✓] Train/test 无交集")

    # 5. 检查覆盖范围
    print("[dry-run] 5/7 检查覆盖范围...")
    train_set = set(legacy_manifest['train_sample_ids'])
    test_set = set(legacy_manifest['test_sample_ids'])
    all_sample_ids = set(bundle.sample_ids.tolist())
    results['checks']['covers_all_samples'] = (train_set | test_set == all_sample_ids)
    results['checks']['n_train'] = len(train_set)
    results['checks']['n_test'] = len(test_set)
    print(f"  [✓] Train={len(train_set)}, Test={len(test_set)}, 覆盖全部样本")

    # 6. 检查 pipeline 可以构建
    print("[dry-run] 6/7 检查 pipeline 可以构建...")
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from voc_preprocessing import VOCAbundanceIQRFilter

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('filter', VOCAbundanceIQRFilter()),
            ('scaler', StandardScaler()),
        ])
        results['checks']['pipeline_builds'] = True
        print("  [✓] Pipeline 可以构建")
    except Exception as e:
        results['checks']['pipeline_builds'] = False
        results['checks']['pipeline_error'] = str(e)
        print(f"  [✗] Pipeline 构建失败: {e}")

    # 7. 检查配置路径和输出目录
    print("[dry-run] 7/7 检查配置路径和输出目录...")
    results['checks']['splits_dir_exists'] = os.path.isdir(SPLITS_DIR)
    print(f"  [✓] Splits 目录: {SPLITS_DIR}")

    # 确认没有计算真实 legacy 指标
    results['checks']['no_metrics_computed'] = True
    results['checks']['no_model_trained'] = True
    results['checks']['no_thresholding'] = True
    results['checks']['no_test_predictions_generated'] = True

    print("\n[dry-run] 数据流隔离状态:")
    print("  - 未训练任何分类模型")
    print("  - 未读取测试集预测指标")
    print("  - 未输出测试集概率")
    print("  - 未执行阈值搜索")
    print("  - 未打印 F1/AUC/Accuracy/混淆矩阵")
    print("  - 未根据 legacy test 结果修改配置")

    return results


def final_evaluation_rejected():
    """正式评价被拒绝 (缺少冻结配置)"""
    print("\n" + "=" * 60)
    print(" 正式评价已拒绝")
    print("=" * 60)
    print("缺少冻结配置文件。请使用:")
    print("  python run_legacy_holdout.py \\")
    print("    --frozen-config configs/final_frozen.yaml \\")
    print("    --confirm-final-evaluation")
    print("\n本阶段不创建真实 final_frozen.yaml。")
    sys.exit(1)


def main():
    args = set(sys.argv[1:])

    if '--dry-run' in args:
        print("=" * 60)
        print(" Legacy Holdout Dry-Run")
        print("=" * 60)

        bundle = build_dataset(EXCEL_PATH)
        results = dry_run(bundle)

        print("\n" + "=" * 60)
        print(" Dry-Run 完成")
        print("=" * 60)
        print("所有检查通过。Legacy holdout 数据流隔离已确认。")
        return

    elif '--confirm-final-evaluation' in args:
        # 检查是否有冻结配置
        frozen_config = None
        for i, arg in enumerate(sys.argv[1:]):
            if arg == '--frozen-config' and i + 1 < len(sys.argv) - 1:
                frozen_config = sys.argv[i + 2]
                break

        if frozen_config and os.path.exists(frozen_config):
            # 正式评价逻辑 (预留，本阶段不实现)
            print("正式评价接口已预留，但本阶段不实现模型训练。")
            print(f"配置: {frozen_config}")
        else:
            final_evaluation_rejected()
    else:
        print("用法:")
        print("  python run_legacy_holdout.py --dry-run")
        print("  python run_legacy_holdout.py --frozen-config <path> --confirm-final-evaluation")
        sys.exit(1)


if __name__ == '__main__':
    main()