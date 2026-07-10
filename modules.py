import torch
import torch.nn as nn
import torch.nn.functional as F


class ParalleResidualBlock(nn.Module):
    def __init__(self, num_clusters, dim=128, hidden_dim=1024):
        super().__init__()
        total_in = num_clusters * dim
        total_hidden = num_clusters * hidden_dim
        self.net = nn.Sequential(
            nn.Conv1d(total_in, total_hidden, kernel_size=1, groups=num_clusters),
            nn.BatchNorm1d(total_hidden),
            nn.ReLU(),
            nn.Conv1d(total_hidden, total_in, kernel_size=1, groups=num_clusters),
            nn.BatchNorm1d(total_in),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


# ======================== 新增分类器 ========================

class CNNClassifier_Seq(nn.Module):
    """路线A: 纯 Conv1d 序列，参数共享特征交叉 + 残差"""
    def __init__(self, input_size, num_class, dropout=0.4):
        super().__init__()
        # Block 1: 1→16, k=7→5
        self.conv1a = nn.Conv1d(1, 16, 7, padding=3)
        self.bn1a = nn.BatchNorm1d(16)
        self.conv1b = nn.Conv1d(16, 16, 5, padding=2)
        self.bn1b = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        # Block 2: 16→32, k=5→3
        self.conv2a = nn.Conv1d(16, 32, 5, padding=2)
        self.bn2a = nn.BatchNorm1d(32)
        self.conv2b = nn.Conv1d(32, 32, 3, padding=1)
        self.bn2b = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        # Block 3: 32→64, k=3→3
        self.conv3a = nn.Conv1d(32, 64, 3, padding=1)
        self.bn3a = nn.BatchNorm1d(64)
        self.conv3b = nn.Conv1d(64, 64, 3, padding=1)
        self.bn3b = nn.BatchNorm1d(64)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # Shared
        self.act = nn.GELU()
        self.drop3 = nn.Dropout(dropout - 0.1)
        self.drop4 = nn.Dropout(dropout)
        self.drop5 = nn.Dropout(dropout + 0.1)
        # Head
        self.head = nn.Sequential(
            nn.Linear(64, 16), nn.GELU(), nn.Dropout(dropout + 0.1),
            nn.Linear(16, num_class)
        )
        # 1x1 conv projections for residual connections
        self.res1 = nn.Conv1d(1, 16, 1)    # block 1: 1→16
        self.res2 = nn.Conv1d(16, 32, 1)   # block 2: 16→32
        self.res3 = nn.Conv1d(32, 64, 1)   # block 3: 32→64

    def _conv_block(self, x, conv_a, bn_a, conv_b, bn_b, drop, residual=None):
        r = residual(x) if residual is not None else x
        out = conv_a(x)
        out = bn_a(out)
        out = self.act(out)
        out = drop(out)
        out = conv_b(out)
        out = bn_b(out)
        return self.act(out + r)

    def forward(self, x):
        # x: [B, D]
        x = x.unsqueeze(1)  # [B, 1, D]
        # Block 1: 1→16
        x = self._conv_block(x, self.conv1a, self.bn1a,
                             self.conv1b, self.bn1b, self.drop3, self.res1)
        x = self.pool1(x)
        # Block 2: 16→32
        x = self._conv_block(x, self.conv2a, self.bn2a,
                             self.conv2b, self.bn2b, self.drop4, self.res2)
        x = self.pool2(x)
        # Block 3: 32→64
        x = self._conv_block(x, self.conv3a, self.bn3a,
                             self.conv3b, self.bn3b, self.drop5, self.res3)
        x = self.gap(x).flatten(1)
        return self.head(x)


class CNNClassifier_Cluster(nn.Module):
    """路线B: 簇感知多路径 CNN"""
    def __init__(self, cluster_mask, num_class, dropout=0.4):
        super().__init__()
        self.register_buffer('cluster_mask', cluster_mask.float())
        self.num_clusters = cluster_mask.shape[0]
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, 5, padding=2),
                nn.BatchNorm1d(16),
                nn.GELU(),
                nn.Dropout(dropout - 0.1),
                nn.Conv1d(16, 16, 5, padding=2),
                nn.BatchNorm1d(16),
                nn.GELU(),
                nn.Dropout(dropout - 0.1),
                nn.AdaptiveAvgPool1d(1),
            ) for _ in range(self.num_clusters)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(self.num_clusters * 16, 32),
            nn.GELU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(32, num_class)
        )

    def forward(self, x):
        outs = []
        for k in range(self.num_clusters):
            x_k = x * self.cluster_mask[k]
            nonzero_idx = self.cluster_mask[k].nonzero(as_tuple=True)[0]
            if len(nonzero_idx) == 0:
                continue
            x_k = x_k[:, nonzero_idx].unsqueeze(1)
            out = self.paths[k](x_k)
            outs.append(out.flatten(1))
        return self.fusion(torch.cat(outs, dim=1))


class DeepMLPClassifier(nn.Module):
    """路线C: 加深 MLP 基线"""
    def __init__(self, input_size, num_class, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(dropout - 0.1),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32),
            nn.ReLU(), nn.Dropout(dropout + 0.1),
            nn.Linear(32, num_class)
        )

    def forward(self, x):
        return self.net(x)


# ======================== MultiView 主模型 ========================

class MultiView(nn.Module):

    def __init__(self, k_mean_mask, num_blocks=8, head_dim=128, num_class=None,
                 temperature=0.5, classifier_type='linear'):
        super().__init__()
        self.num_class = num_class
        self.register_buffer('cluster_mask', k_mean_mask.float())
        self.num_clusters, self.input_size = k_mean_mask.shape
        self.head_dim = head_dim
        self.temperature = temperature                # 训练时可逐 epoch 调

        self.logist = nn.Sequential(
            nn.Conv1d(self.input_size * self.num_clusters,
                      self.head_dim * self.num_clusters,
                      kernel_size=1, groups=self.num_clusters),
            nn.BatchNorm1d(self.head_dim * self.num_clusters),
            nn.ReLU(),
            nn.Sequential(*[ParalleResidualBlock(self.num_clusters, dim=head_dim)
                            for _ in range(num_blocks)]),
            nn.Conv1d(self.head_dim * self.num_clusters,
                      self.input_size * self.num_clusters,
                      kernel_size=1, groups=self.num_clusters),
        )

        # 分类器选择
        if classifier_type == 'cnn_seq':
            self.mlp_classifier = CNNClassifier_Seq(self.input_size, self.num_class)
        elif classifier_type == 'cnn_cluster':
            self.mlp_classifier = CNNClassifier_Cluster(self.cluster_mask, self.num_class)
        elif classifier_type == 'deep_mlp':
            self.mlp_classifier = DeepMLPClassifier(self.input_size, self.num_class)
        else:  # 'linear' 或任何未识别的值
            self.mlp_classifier = nn.Sequential(
                nn.Linear(self.input_size, self.num_class)
            )


    def mask_logist(self, x):
        b = x.shape[0]
        x_split = x.unsqueeze(1) * self.cluster_mask        # [B, K, D]
        x_flat  = x_split.reshape(b, -1, 1)                 # [B, K*D, 1]
        out     = self.logist(x_flat).view(b, self.num_clusters, self.input_size)
        return (out * self.cluster_mask).sum(dim=1)         # [B, D]

    def get_score(self, x):
        """
        训练: 返回带 STE 的二值 mask, 前向 ∈ {0,1}, 反向梯度走 sigmoid
        推理: 返回确定的二值 mask, sigma(theta) > 0.5  <=>  theta > 0
        """
        theta = self.mask_logist(x).mean(dim=0)             # [D]
        if self.training:
            u      = torch.rand_like(theta).clamp_(1e-6, 1 - 1e-6)
            gumbel = torch.log(u) - torch.log1p(-u)         # logistic 噪声
            soft   = torch.sigmoid((theta + gumbel) / self.temperature)
            hard   = (theta + gumbel > 0).float()
            # 前向值 = hard;  反向梯度 = d soft / d theta
            return hard + (soft - soft.detach())
        else:
            return (theta > 0).float()

    def selection_prob(self, x):
        """sigma(theta), 即每个 VOC 被选中的概率, 用于稀疏正则."""
        return torch.sigmoid(self.mask_logist(x).mean(dim=0))

    def forward(self, x):
        mask  = self.get_score(x)
        logit = self.mlp_classifier(x * mask)
        return logit, mask