"""
VOC Dataset 模块

功能：
1. 读取筛选后的VOC Excel
2. 标签转换
3. 保存mat文件
4. 提供PyTorch Dataset

"""


import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from scipy.io import savemat



class VOCDataset(Dataset):

    def __init__(
            self,
            excel_path,
            label_map=None
    ):

        if label_map is None:
            label_map = {
                1: 0,
                2: 0,
                3: 1
            }


        df = pd.read_excel(
            excel_path
        )


        # 标签
        self.y = torch.tensor(
            df["Class"]
            .map(label_map)
            .values,
            dtype=torch.long
        )


        # 特征
        self.X = torch.tensor(
            df.drop(
                columns=["Class"]
            )
            .values,
            dtype=torch.float32
        )


        # VOC名称
        self.feat_names = (
            df.drop(
                columns=["Class"]
            )
            .columns
            .tolist()
        )



    def __len__(self):

        return len(self.y)



    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx]
        )



    def save_mat(
            self,
            output_path
    ):

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )


        savemat(
            output_path,
            {

                "X":
                    self.X.numpy(),


                "y":
                    self.y.numpy()
                    .reshape(-1,1),


                "feat_names":
                    np.array(
                        self.feat_names,
                        dtype=object
                    )

            }
        )


        print(
            f"saved mat: {output_path}"
        )



def create_mat_dataset(
        excel_path,
        mat_path
):

    """
    Excel -> mat

    """

    dataset = VOCDataset(
        excel_path
    )


    print(
        "samples:",
        len(dataset)
    )

    print(
        "features:",
        len(dataset.feat_names)
    )


    dataset.save_mat(
        mat_path
    )


    return dataset



if __name__ == "__main__":


    create_mat_dataset(

        excel_path=
        "data/filtered_voc_step2_with_unknown.xlsx",


        mat_path=
        "data/voc_dataset_1+2_vs_3_with_unknown.mat"

    )