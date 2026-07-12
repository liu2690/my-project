#!/usr/bin/env python3
"""
run_nested_cv.py — 阶段二：传统模型 Nested CV 入口

用法:
  python run_nested_cv.py --quick
  python run_nested_cv.py --full
  python run_nested_cv.py --full --n-jobs 4
  python run_nested_cv.py --dataflow-smoke
  python run_nested_cv.py --freeze-config
"""

import sys
import os

from nested_cv_engine import run_nested_cv, run_dataflow_smoke, _compute_frozen_config_hash
from candidate_registry import get_candidate_counts


def print_help():
    print("""
用法:
  python run_nested_cv.py --quick             快速模式 (仅 outer fold 0,1)
  python run_nested_cv.py --full               完整模式 (全部 5 个 outer folds)
  python run_nested_cv.py --full --n-jobs 4    完整模式 + 并行
  python run_nested_cv.py --dataflow-smoke     数据流 smoke 测试 (不读取 outer-test 标签)
  python run_nested_cv.py --freeze-config      显示 frozen config 哈希

说明:
  --quick           缩小参数网格，仅验证数据流，不是正式结果
  --full            完整参数网格，正式 Nested CV 结果
  --dataflow-smoke  验证 pipeline 数据流，绝不读取 canonical outer-test 标签
  --freeze-config   显示 frozen config SHA-256 哈希
  --n-jobs          并行任务数 (默认 1)
""")
    # 显示候选数
    counts = get_candidate_counts()
    for mode in ['quick', 'full']:
        print(f"\n{mode} 模式候选数: {counts[mode]['total']}")
        for fam, cnt in counts[mode].items():
            if fam != 'total':
                print(f"  {fam}: {cnt}")


def main():
    args = sys.argv[1:]

    if not args or '--help' in args or '-h' in args:
        print_help()
        sys.exit(0)

    # --freeze-config: 显示 frozen config 哈希
    if '--freeze-config' in args:
        try:
            h = _compute_frozen_config_hash()
            print(f"Frozen config hash (SHA-256): {h}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    # --dataflow-smoke: 运行 smoke 测试
    if '--dataflow-smoke' in args:
        print("Running dataflow smoke test...")
        print("(This mode NEVER reads canonical outer-test labels)")
        print()
        run_dataflow_smoke(n_jobs=1)
        return

    mode = None
    n_jobs = 1
    skip_next = False

    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue

        if arg == '--quick':
            if mode is not None:
                print("Error: --quick and --full are mutually exclusive")
                sys.exit(1)
            mode = 'quick'
        elif arg == '--full':
            if mode is not None:
                print("Error: --quick and --full are mutually exclusive")
                sys.exit(1)
            mode = 'full'
        elif arg == '--n-jobs':
            if i + 1 < len(args):
                try:
                    n_jobs = int(args[i + 1])
                    skip_next = True
                except ValueError:
                    print(f"Error: invalid --n-jobs value: {args[i + 1]}")
                    sys.exit(1)
            else:
                print("Error: --n-jobs requires a value")
                sys.exit(1)

    if mode is None:
        print("Error: must specify --quick, --full, --dataflow-smoke, or --freeze-config")
        print_help()
        sys.exit(1)

    counts = get_candidate_counts()
    print(f"\nMode: {mode}")
    print(f"Candidates: {counts[mode]['total']}")
    for fam, cnt in counts[mode].items():
        if fam != 'total':
            print(f"  {fam}: {cnt}")

    if mode == 'quick':
        outer_folds = [0, 1]
    else:
        outer_folds = None  # all 5

    print(f"Outer folds: {outer_folds if outer_folds else 'all (0-4)'}")
    print(f"n_jobs: {n_jobs}")
    print()

    run_nested_cv(
        mode=mode,
        n_jobs=n_jobs,
        outer_folds=outer_folds,
    )


if __name__ == '__main__':
    main()