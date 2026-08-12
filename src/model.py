"""
MultiView VOC 特征选择模型

包含:
    ParalleResidualBlock
    MultiView

"""

import torch

import torch.nn as nn



class ParalleResidualBlock(nn.Module):

    def __init__(
            self,
            num_clusters,
            dim=128,
            hidden_dim=1024
    ):

        super().__init__()


        total_in = (
            num_clusters
            *
            dim
        )


        total_hidden = (
            num_clusters
            *
            hidden_dim
        )


        self.net = nn.Sequential(

            nn.Conv1d(
                total_in,
                total_hidden,
                kernel_size=1,
                groups=num_clusters
            ),


            nn.BatchNorm1d(
                total_hidden
            ),


            nn.ReLU(),


            nn.Conv1d(
                total_hidden,
                total_in,
                kernel_size=1,
                groups=num_clusters
            ),


            nn.BatchNorm1d(
                total_in
            )

        )


        self.relu = nn.ReLU()



    def forward(self,x):

        return self.relu(
            self.net(x)+x
        )

class CNNClassifier(nn.Module):

    def __init__(
            self,
            input_size,
            num_classes=2,
            channel_1=8,
            channel_2=16,
            pooled_length=4,
            dropout=0.40
    ):
        super().__init__()

        self.pooled_length = pooled_length

        self.features = nn.Sequential(

            # [B, 1, F] -> [B, 16, F]
            nn.Conv1d(
                in_channels=1,
                out_channels=channel_1,
                kernel_size=5,
                padding=2,
                bias=False
            ),

            nn.GroupNorm(
                num_groups=4,
                num_channels=channel_1
            ),

            nn.GELU(),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2
            ),


            # [B, 16, F/2] -> [B, 32, F/2]
            nn.Conv1d(
                in_channels=channel_1,
                out_channels=channel_2,
                kernel_size=5,
                padding=2,
                bias=False
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=channel_2
            ),

            nn.GELU(),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2
            ),

            nn.AdaptiveAvgPool1d(
                output_size=pooled_length
            )
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                channel_2 * pooled_length,
                32
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                32,
                num_classes
            )
        )


    def forward(self, x):

        # x:
        # [B, F]

        x = x.unsqueeze(1)

        # [B, 1, F]

        x = self.features(x)

        x = self.classifier(x)

        return x



class MultiView(nn.Module):

    def __init__(
            self,
            cluster_mask,
            num_blocks=8,
            head_dim=128,
            num_classes=2,
            temperature=0.5
    ):

        super().__init__()


        self.register_buffer(
            "cluster_mask",
            cluster_mask.float()
        )


        self.num_clusters, self.input_size = (
            cluster_mask.shape
        )


        self.head_dim = head_dim

        self.temperature = temperature



        self.logist = nn.Sequential(

            nn.Conv1d(
                self.input_size*self.num_clusters,
                self.head_dim*self.num_clusters,
                kernel_size=1,
                groups=self.num_clusters
            ),


            nn.BatchNorm1d(
                self.head_dim*self.num_clusters
            ),


            nn.ReLU(),


            *[
                ParalleResidualBlock(
                    self.num_clusters,
                    dim=head_dim
                )

                for _ in range(num_blocks)
            ],


            nn.Conv1d(
                self.head_dim*self.num_clusters,
                self.input_size*self.num_clusters,
                kernel_size=1,
                groups=self.num_clusters
            )
        )



        self.classifier = CNNClassifier(

            input_size=self.input_size,

            num_classes=num_classes,

            channel_1=16,

            channel_2=32,

            pooled_length=8,

            dropout=0.30
        )



    def mask_logist(self,x):

        batch_size=x.shape[0]


        x_split = (
            x.unsqueeze(1)
            *
            self.cluster_mask
        )


        x_flat = x_split.reshape(
            batch_size,
            -1,
            1
        )


        out=self.logist(
            x_flat
        )


        out=out.view(
            batch_size,
            self.num_clusters,
            self.input_size
        )


        return (
            out
            *
            self.cluster_mask
        ).sum(dim=1)





    def get_score(self,x):

        theta = (
            self.mask_logist(x)
            .mean(dim=0)
        )


        if self.training:

            u=torch.rand_like(theta)

            u=u.clamp(
                1e-6,
                1-1e-6
            )


            gumbel = (
                torch.log(u)
                -
                torch.log1p(-u)
            )


            soft=torch.sigmoid(
                (theta+gumbel)
                /
                self.temperature
            )


            hard=(
                theta+gumbel>0
            ).float()


            return (
                hard
                +
                soft
                -
                soft.detach()
            )


        else:

            return (
                theta>0
            ).float()



    def selection_prob(self,x):

        return torch.sigmoid(
            self.mask_logist(x)
            .mean(dim=0)
        )



    def forward(self, x):

        mask = self.get_score(x)

        selected_x = x * mask

        logits = self.classifier(
            selected_x
        )

        return logits, mask