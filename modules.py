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


class MultiView(nn.Module):

    def __init__(self, k_mean_mask, num_blocks=8, head_dim=128, num_class=None,
                 temperature=0.5):
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