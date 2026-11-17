
import torch
import torch.nn as nn


# ============================================================
# 1. BPD 原有并行残差模块
# ============================================================

class ParalleResidualBlock(nn.Module):

    def __init__(
        self,
        num_clusters,
        dim=128,
        hidden_dim=1024
    ):
        super().__init__()

        total_in = num_clusters * dim
        total_hidden = num_clusters * hidden_dim

        self.net = nn.Sequential(
            nn.Conv1d(
                total_in,
                total_hidden,
                kernel_size=1,
                groups=num_clusters
            ),
            nn.BatchNorm1d(total_hidden),
            nn.ReLU(),

            nn.Conv1d(
                total_hidden,
                total_in,
                kernel_size=1,
                groups=num_clusters
            ),
            nn.BatchNorm1d(total_in),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + x)


# ============================================================
# 2. CNN 分类器
# ============================================================

class CNNClassifier(nn.Module):

    def __init__(
        self,
        num_class=2,
        channels=(16, 32, 64),
        kernel_size=7,
        dropout=0.30
    ):
        super().__init__()

        c1, c2, c3 = channels
        padding = kernel_size // 2

        self.features = nn.Sequential(

            # Conv Block 1
            nn.Conv1d(
                1,
                c1,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.GroupNorm(4, c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Conv Block 2
            nn.Conv1d(
                c1,
                c2,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.GroupNorm(8, c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Conv Block 3
            nn.Conv1d(
                c2,
                c3,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.GroupNorm(8, c3),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(c3 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_class)
        )

    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError(
                f"CNN 输入形状错误: {x.shape}"
            )

        x = self.features(x)

        avg_features = self.avg_pool(x).squeeze(-1)
        max_features = self.max_pool(x).squeeze(-1)

        features = torch.cat(
            [avg_features, max_features],
            dim=1
        )

        return self.classifier(features)


# ============================================================
# 3. BPD MultiView + CNN
# ============================================================

class MultiView(nn.Module):

    def __init__(
        self,
        k_mean_mask,
        num_blocks=8,
        head_dim=128,
        num_class=2,
        temperature=0.5
    ):
        super().__init__()

        self.num_class = num_class

        self.register_buffer(
            'cluster_mask',
            k_mean_mask.float()
        )

        self.num_clusters, self.input_size = (
            k_mean_mask.shape
        )

        self.head_dim = head_dim
        self.temperature = temperature

        # ----------------------------------------------------
        # 原有 BPD 特征选择网络
        # 保持原结构不变
        # ----------------------------------------------------

        self.logist = nn.Sequential(

            nn.Conv1d(
                self.input_size * self.num_clusters,
                self.head_dim * self.num_clusters,
                kernel_size=1,
                groups=self.num_clusters
            ),

            nn.BatchNorm1d(
                self.head_dim * self.num_clusters
            ),

            nn.ReLU(),

            nn.Sequential(
                *[
                    ParalleResidualBlock(
                        self.num_clusters,
                        dim=head_dim
                    )
                    for _ in range(num_blocks)
                ]
            ),

            nn.Conv1d(
                self.head_dim * self.num_clusters,
                self.input_size * self.num_clusters,
                kernel_size=1,
                groups=self.num_clusters
            ),
        )

        # ----------------------------------------------------
        # CNN 替换原 MLP 分类器
        # ----------------------------------------------------

        self.cnn_classifier = CNNClassifier(
            num_class=num_class,
            channels=(16, 32, 64),
            kernel_size=7,
            dropout=0.30
        )

    # ========================================================
    # 特征选择网络
    # ========================================================

    def mask_logist(self, x):

        b = x.shape[0]

        x_split = (
            x.unsqueeze(1)
            * self.cluster_mask
        )

        x_flat = x_split.reshape(
            b,
            -1,
            1
        )

        out = self.logist(x_flat).view(
            b,
            self.num_clusters,
            self.input_size
        )

        return (
            out * self.cluster_mask
        ).sum(dim=1)

    # ========================================================
    # Gumbel-Softmax 特征选择
    # ========================================================

    def get_score(self, x):

        theta = self.mask_logist(x).mean(dim=0)

        if self.training:

            u = torch.rand_like(theta).clamp_(
                1e-6,
                1 - 1e-6
            )

            gumbel = (
                torch.log(u)
                - torch.log1p(-u)
            )

            soft = torch.sigmoid(
                (theta + gumbel) / self.temperature
            )

            hard = (
                theta + gumbel > 0
            ).float()

            return hard + (
                soft - soft.detach()
            )

        return (
            theta > 0
        ).float()

    # ========================================================
    # 特征选择概率
    # ========================================================

    def selection_prob(self, x):

        return torch.sigmoid(
            self.mask_logist(x).mean(dim=0)
        )

    # ========================================================
    # 前向传播
    # ========================================================

    def forward(self, x):

        mask = self.get_score(x)

        x_masked = x * mask

        logits = self.cnn_classifier(
            x_masked.unsqueeze(1)
        )

        return logits, mask
