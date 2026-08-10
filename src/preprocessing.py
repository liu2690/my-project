"""
VOC 数据预处理模块

功能:
1. 读取 1、2、3.xlsx
2. 控制是否保留 Unknown VOC
3. 丰度筛选
4. IQR筛选
5. 保存筛选后的Excel
6. 生成mat数据集

"""

import os
import numpy as np
import pandas as pd

from scipy.io import savemat


# ============================================================
# 1. Unknown控制接口
# ============================================================

def process_unknown(
        all_feature_id,
        all_feature_name,
        data_raw,
        use_unknown=True
):
    """
    控制是否保留Unknown VOC

    Parameters
    ----------
    use_unknown:
        True:
            保留Unknown

        False:
            删除Unknown

    """

    if use_unknown:

        return (
            all_feature_id,
            all_feature_name,
            data_raw
        )


    keep = all_feature_name != "Unknown"


    return (
        all_feature_id[keep],
        all_feature_name[keep],
        data_raw[:, keep]
    )



# ============================================================
# 2. 丰度筛选
# ============================================================

def abundance_filter(
        data,
        feature_id,
        feature_name,
        percentile=40
):

    mean_data = np.mean(
        data,
        axis=0
    )


    threshold = np.percentile(
        mean_data,
        percentile
    )


    mask = mean_data > threshold


    return (
        data[:, mask],
        feature_id[mask],
        feature_name[mask],
        threshold
    )



# ============================================================
# 3. IQR筛选
# ============================================================

def iqr_filter(
        data,
        feature_id,
        feature_name,
        percentile=25
):

    log_data = np.log1p(data)


    iqr = (
        np.percentile(
            log_data,
            75,
            axis=0
        )
        -
        np.percentile(
            log_data,
            25,
            axis=0
        )
    )


    threshold = np.percentile(
        iqr,
        percentile
    )


    mask = iqr >= threshold


    return (
        log_data[:, mask],
        feature_id[mask],
        feature_name[mask],
        threshold
    )



# ============================================================
# 4. 主预处理函数
# ============================================================

def preprocess_voc(
        input_file,
        output_excel,
        use_unknown=True
):

    """
    VOC完整预处理流程

    Parameters
    ----------

    input_file:
        原始Excel

    output_excel:
        筛选后Excel

    use_unknown:
        是否保留Unknown

    """


    # ------------------------
    # 读取数据
    # ------------------------

    df = pd.read_excel(
        input_file
    )


    feature_id = np.array(
        df.columns[1:].tolist()
    )


    feature_name = np.array(
        df.iloc[0,1:].values
    )


    data_raw = (
        df.iloc[1:,1:]
        .values
        .astype(np.float64)
    )


    classes = (
        df.iloc[1:,0]
        .values
        .astype(int)
    )


    print("="*60)

    print(
        "原始特征:",
        data_raw.shape[1]
    )


    # ------------------------
    # Unknown处理
    # ------------------------

    (
        feature_id,
        feature_name,
        data_raw

    ) = process_unknown(
        feature_id,
        feature_name,
        data_raw,
        use_unknown
    )


    print(
        "Unknown模式:",
        use_unknown
    )


    print(
        "Unknown处理后:",
        data_raw.shape[1]
    )



    # ------------------------
    # 丰度筛选
    # ------------------------

    (
        data_step1,
        fid_step1,
        fname_step1,
        abundance_threshold

    ) = abundance_filter(
        data_raw,
        feature_id,
        feature_name
    )


    print(
        "丰度筛选后:",
        data_step1.shape[1]
    )


    # ------------------------
    # IQR筛选
    # ------------------------

    (
        data_step2,
        fid_step2,
        fname_step2,
        iqr_threshold

    ) = iqr_filter(
        data_step1,
        fid_step1,
        fname_step1
    )


    print(
        "IQR筛选后:",
        data_step2.shape[1]
    )



    # ------------------------
    # 保存Excel
    # ------------------------

    columns = [
        f"{fid}_{name}"
        for fid,name
        in zip(
            fid_step2,
            fname_step2
        )
    ]


    df_out = pd.DataFrame(
        data_step2,
        columns=columns
    )


    df_out.insert(
        0,
        "Class",
        classes
    )


    df_out.to_excel(
        output_excel,
        index=False
    )


    print(
        "保存:",
        output_excel
    )


    print("="*60)



    return {

        "feature_names":
            fname_step2,

        "feature_ids":
            fid_step2,

        "classes":
            classes,

        "num_features":
            len(fname_step2)

    }



# ============================================================
# 5. Excel生成mat
# ============================================================

def excel_to_mat(
        excel_file,
        mat_file
):

    df = pd.read_excel(
        excel_file
    )


    X = (
        df.drop(
            columns=["Class"]
        )
        .values
        .astype(np.float32)
    )


    y = (
        df["Class"]
        .values
        .astype(np.int64)
    )


    feature_names = (
        df.drop(
            columns=["Class"]
        )
        .columns
        .tolist()
    )


    savemat(
        mat_file,
        {

            "X":X,

            "y":
                y.reshape(-1,1),

            "feat_names":
                np.array(
                    feature_names,
                    dtype=object
                )

        }
    )


    print(
        "保存mat:",
        mat_file
    )



# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":


    preprocess_voc(

        input_file=
        "data/1、2、3.xlsx",

        output_excel=
        "data/filtered_voc_step2_with_unknown.xlsx",

        use_unknown=True
    )


    excel_to_mat(

        "data/filtered_voc_step2_with_unknown.xlsx",

        "data/voc_dataset_1+2_vs_3_with_unknown.mat"

    )