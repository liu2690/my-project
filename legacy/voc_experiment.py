import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib
from legacy import modules
importlib.reload(modules)
from legacy.utils import *
from legacy.modules import *
import copy
import os
import json
import random
import scipy.io as sio
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """设置随机种子，cudnn.benchmark=True 加速 CNN 计算"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def compute_metrics(cm, targets, probs):
    tn, fp = float(cm[0, 0]), float(cm[0, 1])
    fn, tp = float(cm[1, 0]), float(cm[1, 1])
    eps = 1e-12
    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    ppv = tp / (tp + fp + eps)
    npv = tn / (tn + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = 2 * ppv * sens / (ppv + sens + eps)
    try:
        auc_val = roc_auc_score(targets, probs)
    except ValueError:
        auc_val = np.nan
    return {'Sensitivity': sens, 'Specificity': spec, 'PPV': ppv, 'NPV': npv,
            'Accuracy': acc, 'F1': f1, 'AUC': auc_val}


def mean_ci(values, alpha=0.05):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    m = float(v.mean()) if n > 0 else np.nan
    if n < 2:
        return m, np.nan, np.nan, np.nan, np.nan
    sd = float(v.std(ddof=1))
    sem = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return m, sd, sem, m - tcrit * sem, m + tcrit * sem


def experiment(data_path, save_path, num_cluster,
               experiment_repeats=2, repeats=2, epochs=200,
               sparsity_lambda=1e-2,
               temp_start=1.0, temp_end=0.3,
               seed=42, panel_threshold=0.5,
               classifier_type='linear',
               record_train_f1=False):

    set_seed(seed)

    out_dir = os.path.dirname(save_path)
    os.makedirs(out_dir, exist_ok=True)

    data = sio.loadmat(data_path)
    samples = torch.tensor(data['X'], dtype=DTYPE)
    labels = (torch.tensor(data['y'], dtype=DTYPE).view(-1)).long()
    voc_names = [str(v.flat[0]) for v in data['feat_names'].flatten()]

    train_loader, test_loader, _ = split_dataset(
        samples, labels, batch_size=32, split_length=[0.8, 0.2])
    _tmp = DataLoader(train_loader.dataset, batch_size=len(train_loader.dataset), shuffle=False)
    x_pool, y_pool = next(iter(_tmp))

    k_means_mask = feature_cluster(x_pool.float().numpy(), num_cluster)

    # 类别不平衡补偿：计算 train 集上的类别权重
    class_counts = torch.bincount(y_pool.long())
    class_weights = torch.tensor(
        [1.0, class_counts[0].item() / max(class_counts[1].item(), 1e-6)],
        device=DEVICE
    )

    all_masks = []
    all_soft = []
    all_cm = []
    all_probs = []
    all_targets = []
    all_metrics = []
    all_train_f1 = []  # Phase 1 诊断用

    pbar = tqdm(range(experiment_repeats))
    for out_idx in pbar:

        train_loader_init, val_loader, x_train = split_dataset(
            x_pool, y_pool, batch_size=32, split_length=[0.75, 0.25])
        train_loader_init = torch.utils.data.DataLoader(
            train_loader_init.dataset, batch_size=32, shuffle=True, drop_last=True)
        x_val = torch.cat([x for x, _ in val_loader])

        group_best_acc, group_best_mask, group_best_wts = -1.0, None, None

        for in_idx in range(repeats):
            model = MultiView(
                k_mean_mask=k_means_mask,
                num_blocks=4,
                head_dim=256,
                num_class=2,
                temperature=temp_start,
                classifier_type=classifier_type,
            ).to(DEVICE, dtype=DTYPE)

            # 分层学习率：logist 低 lr，classifier 高 lr
            optimizer = optim.AdamW([
                {'params': model.logist.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},
                {'params': model.mlp_classifier.parameters(), 'lr': 5e-5, 'weight_decay': 1e-2},
            ])
            criterion = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=0.05
            )
            local_best_acc, local_best_wts, local_best_mask = 0.0, None, None

            for epoch in range(epochs):

                model.temperature = temp_start + (temp_end - temp_start) * (epoch / max(1, epochs - 1))

                model.train()
                for x, y in train_loader_init:
                    x, y = x.to(DEVICE, dtype=DTYPE), y.to(DEVICE).long()
                    optimizer.zero_grad()
                    outputs, _, sparsity = model(x)
                    loss = criterion(outputs, y) + sparsity_lambda * sparsity
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                model.eval()
                correct, total_n = 0, 0
                with torch.no_grad():
                    mask_eval = model.get_score(x_val.to(DEVICE, dtype=DTYPE))
                    for x, y in val_loader:
                        x, y = x.to(DEVICE, dtype=DTYPE), y.to(DEVICE).long()
                        outputs = model.mlp_classifier(x * mask_eval)
                        correct += (outputs.argmax(1) == y).sum().item()
                        total_n += y.size(0)
                    acc = correct / total_n

                if acc > local_best_acc:
                    local_best_acc = acc
                    local_best_wts = copy.deepcopy(model.state_dict())
                    local_best_mask = mask_eval.cpu()

            if local_best_acc > group_best_acc:
                group_best_acc = local_best_acc
                group_best_wts = local_best_wts
                group_best_mask = local_best_mask

            # Phase 1 诊断：记录 train F1（使用最佳 val 模型）
            if record_train_f1 and group_best_wts is not None:
                model.load_state_dict(group_best_wts)
                model.eval()
                with torch.no_grad():
                    mask_eval = model.get_score(x_val.to(DEVICE, dtype=DTYPE))
                    train_correct, train_total = 0, 0
                    train_preds_all, train_labels_all = [], []
                    for x_t, y_t in train_loader_init:
                        x_t = x_t.to(DEVICE, dtype=DTYPE)
                        y_t = y_t.to(DEVICE).long()
                        outputs = model.mlp_classifier(x_t * mask_eval)
                        preds = outputs.argmax(1)
                        train_correct += (preds == y_t).sum().item()
                        train_total += y_t.size(0)
                        train_preds_all.extend(preds.cpu().numpy())
                        train_labels_all.extend(y_t.cpu().numpy())
                    train_acc = train_correct / train_total
                    # 计算 train F1
                    train_cm = confusion_matrix(train_labels_all, train_preds_all, labels=[0, 1])
                    tn, fp = float(train_cm[0, 0]), float(train_cm[0, 1])
                    fn, tp = float(train_cm[1, 0]), float(train_cm[1, 1])
                    eps = 1e-12
                    train_sens = tp / (tp + fn + eps)
                    train_ppv = tp / (tp + fp + eps)
                    train_f1 = 2 * train_sens * train_ppv / (train_sens + train_ppv + eps)
                    all_train_f1.append({
                        'Repeat': out_idx, 'Train_Acc': train_acc, 'Train_F1': train_f1
                    })

        if group_best_wts is not None:
            model.load_state_dict(group_best_wts)
            model.eval()
            with torch.no_grad():
                m_final = group_best_mask.to(DEVICE, dtype=DTYPE)

                soft_score = model.selection_prob(x_val.to(DEVICE, dtype=DTYPE)).cpu().numpy()

                g_preds, g_probs, g_targets = [], [], []
                for x_test, y_test in test_loader:
                    x_test = x_test.to(DEVICE, dtype=DTYPE)
                    outputs = model.mlp_classifier(x_test * m_final)
                    probs = torch.softmax(outputs.float(), dim=1)[:, 1]
                    preds = outputs.argmax(1)
                    g_preds.extend(preds.cpu().numpy())
                    g_probs.extend(probs.cpu().numpy())
                    g_targets.extend(y_test.cpu().numpy())

                group_best_cm = confusion_matrix(g_targets, g_preds, labels=[0, 1])

            g_probs = np.array(g_probs)
            g_targets = np.array(g_targets)

            metrics = compute_metrics(group_best_cm, g_targets, g_probs)
            metrics['Repeat'] = out_idx
            all_metrics.append(metrics)

            ckpt = {
                'state_dict': group_best_wts,
                'binary_mask': group_best_mask,
                'soft_score': torch.from_numpy(soft_score),
                'val_acc': group_best_acc,
                'seed': seed,
            }
            torch.save(ckpt, os.path.join(out_dir, f'champion_model_repeat{out_idx}.pt'))

            all_masks.append(group_best_mask)
            all_soft.append(soft_score)
            all_cm.append(torch.from_numpy(group_best_cm).unsqueeze(0).float())
            all_probs.append(g_probs)
            all_targets.append(g_targets)

    print(f"[保存] {experiment_repeats} 个冠军模型权重 → {out_dir}/champion_model_repeat*.pt")

    all_masks = torch.stack(all_masks)
    all_soft = np.stack(all_soft)

    selection_freq = all_masks.mean(dim=0).float().cpu().numpy()
    soft_mean = all_soft.mean(axis=0)
    soft_std = all_soft.std(axis=0)
    soft_sem = soft_std / np.sqrt(experiment_repeats)
    combined_cm = torch.cat(all_cm, dim=0)

    sorted_idx = np.lexsort((soft_mean, selection_freq))[::-1]
    top_n = int((selection_freq > panel_threshold).sum())
    if top_n == 0:
        top_n = min(20, len(soft_mean))
        sorted_idx = np.argsort(soft_mean)[::-1]
    top_idx = sorted_idx[:top_n]

    torch.save(all_masks, save_path)
    print(f"[保存①] 原始二值 mask → {save_path}")

    config = {
        'seed': seed, 'num_cluster': num_cluster,
        'experiment_repeats': experiment_repeats, 'repeats': repeats, 'epochs': epochs,
        'batch_size': 32, 'split_outer': [0.8, 0.2], 'split_inner': [0.75, 0.25],
        'num_blocks': 4, 'head_dim': 256,
        'lr_logist': 1e-5, 'lr_classifier': 5e-5,
        'wd_logist': 1e-4, 'wd_classifier': 1e-2,
        'sparsity_lambda': sparsity_lambda,
        'temp_start': temp_start, 'temp_end': temp_end,
        'panel_threshold': panel_threshold,
        'classifier_type': classifier_type,
        'label_smoothing': 0.05, 'class_weight': [1.0, float(class_counts[0] / max(class_counts[1], 1e-6))],
        'grad_clip': 1.0,
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
    }
    with open(os.path.join(out_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[保存②] 运行配置 → {os.path.join(out_dir, 'run_config.json')}")

    # Phase 1 诊断：保存 train F1
    if record_train_f1 and all_train_f1:
        train_f1_df = pd.DataFrame(all_train_f1)
        train_f1_csv = os.path.join(out_dir, 'train_f1_diagnosis.csv')
        train_f1_df.to_csv(train_f1_csv, index=False)
        print(f"[保存②b] Train F1 诊断 → {train_f1_csv}")
        print(f"  Train F1 mean: {train_f1_df['Train_F1'].mean():.4f} ± {train_f1_df['Train_F1'].std():.4f}")

    feat_csv = os.path.join(out_dir, 'feature_selection_stats.csv')
    pd.DataFrame({
        'VOC_Index': np.arange(len(soft_mean)),
        'VOC_Name': voc_names,
        'Selection_Freq': selection_freq,
        'Soft_Mean': soft_mean,
        'Soft_Std': soft_std,
        'Soft_SEM': soft_sem,
    }).to_csv(feat_csv, index=False)
    print(f"[保存③] 特征选择统计 → {feat_csv}")

    sel = selection_freq > panel_threshold
    panel_df = pd.DataFrame({
        'VOC_Index': np.arange(len(soft_mean))[sel],
        'VOC_Name': [voc_names[i] for i in np.where(sel)[0]],
        'Selection_Freq': selection_freq[sel],
        'Soft_Mean': soft_mean[sel],
        'Soft_Std': soft_std[sel],
    }).sort_values(['Selection_Freq', 'Soft_Mean'], ascending=False).reset_index(drop=True)
    panel_csv = os.path.join(out_dir, 'selected_voc_panel.csv')
    panel_df.to_csv(panel_csv, index=False)
    print(f"[保存④] 入选 VOC panel（freq>{panel_threshold}，共 {len(panel_df)} 个）→ {panel_csv}")

    cm_mean = combined_cm.mean(dim=0).numpy()
    pd.DataFrame(cm_mean, index=['True_Neg', 'True_Pos'],
                 columns=['Pred_Neg', 'Pred_Pos']).to_csv(
        os.path.join(out_dir, 'confusion_matrix_summary.csv'))
    print(f"[保存⑤] 混淆矩阵均值 → {os.path.join(out_dir, 'confusion_matrix_summary.csv')}")

    metric_keys = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy', 'F1', 'AUC']
    per_repeat_df = pd.DataFrame(all_metrics)[['Repeat'] + metric_keys]
    per_repeat_df.to_csv(os.path.join(out_dir, 'metrics_per_repeat.csv'), index=False)
    print(f"[保存⑥a] 逐组诊断指标 → {os.path.join(out_dir, 'metrics_per_repeat.csv')}")

    summary_rows = []
    for k in metric_keys:
        m, sd, se, lo, hi = mean_ci(per_repeat_df[k].values)
        summary_rows.append({'Metric': k, 'Mean': m, 'Std': sd, 'SEM': se,
                             'CI95_Lower': lo, 'CI95_Upper': hi})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, 'metrics_summary.csv'), index=False)
    print(f"[保存⑥b] 诊断指标汇总（均值±95%CI）→ {os.path.join(out_dir, 'metrics_summary.csv')}")

    np.savez(os.path.join(out_dir, 'raw_probs_targets.npz'),
             probs=np.array(all_probs, dtype=object),
             targets=np.array(all_targets, dtype=object))
    print(f"[保存⑦] 原始软概率与标签 → {os.path.join(out_dir, 'raw_probs_targets.npz')}")

    mean_fpr = np.linspace(0, 1, 200)
    tprs, aucs = [], []
    for probs_i, targets_i in zip(all_probs, all_targets):
        fpr, tpr, _ = roc_curve(targets_i, probs_i)
        aucs.append(auc(fpr, tpr))
        tpr_i = np.interp(mean_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = tprs.std(axis=0)
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    np.savez(os.path.join(out_dir, 'plot_data.npz'),
             voc_names=np.array(voc_names, dtype=object),
             selection_freq=selection_freq,
             soft_mean=soft_mean, soft_std=soft_std, soft_sem=soft_sem,
             top_idx=top_idx, top_n=top_n,
             mean_fpr=mean_fpr, tprs=tprs, mean_tpr=mean_tpr, std_tpr=std_tpr,
             aucs=np.array(aucs), mean_auc=mean_auc, std_auc=std_auc,
             cm_mean=cm_mean)
    print(f"[保存⑧] 画图用数据 → {os.path.join(out_dir, 'plot_data.npz')}")

    # 保存⑨ 逐组 AUC → CSV
    pd.DataFrame({'Group': np.arange(len(aucs)), 'AUC': aucs}).to_csv(
        os.path.join(out_dir, 'auc_per_group.csv'), index=False)
    print(f"[保存⑨] 逐组AUC → {os.path.join(out_dir, 'auc_per_group.csv')}")

    # 图① 特征重要性
    fig1, axes = plt.subplots(2, 1, figsize=(16, 8))
    axes[0].bar(range(top_n), selection_freq[top_idx], color='teal', alpha=0.7)
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='Threshold 0.5')
    axes[0].set_title(f"Top {top_n} VOC Selection Frequency (n={experiment_repeats} groups)")
    axes[0].set_ylabel("Selection Frequency")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    m_plot, s_plot = soft_mean[top_idx], soft_sem[top_idx]
    axes[1].bar(range(top_n), m_plot, color='coral', alpha=0.6, label='Soft Importance σ(θ)')
    axes[1].fill_between(range(top_n), m_plot - s_plot, m_plot + s_plot,
                         color='gray', alpha=0.4, label='±SEM')
    axes[1].errorbar(range(top_n), m_plot, yerr=soft_std[top_idx],
                     fmt='none', ecolor='black', capsize=2, alpha=0.5, label='Std Dev')
    axes[1].set_title(f"Soft Importance of Top {top_n} Features (Ordered by Selection Freq)")
    axes[1].set_ylabel("σ(θ)")
    axes[1].set_xlabel("VOC Index (Ranked)")
    axes[1].legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"[保存⑩] 特征重要性图 → {os.path.join(out_dir, 'feature_importance.png')}")

    # 图② ROC 曲线
    fig2, ax = plt.subplots(figsize=(7, 7))
    for tpr_i in tprs:
        ax.plot(mean_fpr, tpr_i, color='steelblue', alpha=0.15, linewidth=0.8)
    ax.plot(mean_fpr, mean_tpr, color='navy', linewidth=2,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1),
                    color='steelblue', alpha=0.2, label='±1 SD')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(f'ROC Curve (n={experiment_repeats} champion models, TEST set)', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"[保存⑪] ROC 曲线 → {os.path.join(out_dir, 'roc_curve.png')}")

    # 打印最终指标
    print("\n" + "=" * 60)
    print(f" 🌟 Final Report: averaged over {experiment_repeats} champion models (TEST set)")
    print("-" * 60)
    print(f"{'Metric':<14}{'Mean':>8}{'95% CI':>22}")
    for _, row in summary_df.iterrows():
        print(f"{row['Metric']:<14}{row['Mean']:>8.3f}"
              f"{f'[{row.CI95_Lower:.3f}, {row.CI95_Upper:.3f}]':>22}")
    print("-" * 60)
    print_classification_result(combined_cm)
    print("=" * 60)