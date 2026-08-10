"""
多次重复实验入口

功能:

1. Unknown数据接口
2. K-means k=8
3. train/val/test
4. 多次repeat
5. TEST最终评价
6. 保存统计结果

"""


import os
import json
import random

import numpy as np
import pandas as pd

import torch
import scipy.io as sio


from src.split import split_dataset
from src.clustering import feature_cluster
from src.model import MultiView
from src.trainer import train_model
from src.evaluator import evaluate_model



# ==================================================
# seed
# ==================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic=True

    torch.backends.cudnn.benchmark=False



# ==================================================
# 单次实验
# ==================================================

def run_single_experiment(

        X,

        y,

        repeat_seed,

        num_cluster,

        epochs,

        batch_size

):


    set_seed(
        repeat_seed
    )


    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )


    # -------------------------
    # split
    # -------------------------

    (
        train_loader,
        val_loader,
        test_loader,
        x_train

    ) = split_dataset(

        X,

        y,

        batch_size=batch_size,

        seed=repeat_seed

    )


    # -------------------------
    # Kmeans
    # -------------------------

    cluster_mask = feature_cluster(

        x_train.numpy(),

        num_cluster=num_cluster,

        seed=repeat_seed

    )


    # -------------------------
    # model
    # -------------------------

    model=MultiView(

        cluster_mask=cluster_mask,

        num_blocks=8,

        head_dim=128,

        num_classes=2,

        temperature=1.0

    )


    # -------------------------
    # train
    # -------------------------

    (
        best_state,

        best_mask,

        best_val_acc

    ) = train_model(

        model,

        train_loader,

        val_loader,

        epochs=epochs,

        device=device

    )


    # 加载最佳模型

    model.load_state_dict(
        best_state
    )


    # -------------------------
    # test
    # -------------------------

    test_result=evaluate_model(

        model,

        test_loader,

        device=device

    )


    test_result["Val_Accuracy"]=(
        best_val_acc
    )


    test_result["Seed"]=(
        repeat_seed
    )


    return (

        test_result,

        {

        "state_dict":
            best_state,

        "mask":
            best_mask,

        "cluster_mask":
            cluster_mask,

        "val_acc":
            best_val_acc

        }

    )



# ==================================================
# 总实验
# ==================================================

def experiment(

        data_path=None,

        save_path="result/default",

        use_unknown=True,

        num_cluster=8,

        epochs=30,

        batch_size=16,

        experiment_repeats=100,

        seed=42

):


    os.makedirs(
        save_path,
        exist_ok=True
    )


    model_dir=os.path.join(
        save_path,
        "models"
    )


    os.makedirs(
        model_dir,
        exist_ok=True
    )


    # -------------------------
    # 自动选择数据
    # -------------------------

    if data_path is None:


        if use_unknown:

            data_path=(
                "data/"
                "voc_dataset_1+2_vs_3_with_unknown.mat"
            )

        else:

            data_path=(
                "data/"
                "voc_dataset_1+2_vs_3.mat"
            )


    data=sio.loadmat(
        data_path
    )


    X=torch.tensor(
        data["X"],
        dtype=torch.float32
    )


    y=torch.tensor(
        data["y"].reshape(-1),
        dtype=torch.long
    )


    print("="*60)

    print(
        "Experiment start"
    )

    print(
        "Samples:",
        X.shape[0]
    )

    print(
        "Features:",
        X.shape[1]
    )

    print(
        "Unknown:",
        use_unknown
    )

    print(
        "Repeats:",
        experiment_repeats
    )



    all_results=[]


    # =================================================
    # repeat循环
    # =================================================

    for repeat in range(
        experiment_repeats
    ):


        print(
            "\n"
            +
            "="*50
        )


        print(
            f"Repeat {repeat+1}/{experiment_repeats}"
        )


        result, checkpoint = run_single_experiment(

            X,

            y,

            repeat_seed=
                seed+repeat,

            num_cluster=
                num_cluster,

            epochs=
                epochs,

            batch_size=
                batch_size

        )


        all_results.append(
            result
        )


        torch.save(

            checkpoint,

            os.path.join(

                model_dir,

                f"best_model_repeat{repeat}.pt"

            )

        )


        print(
            result
        )



    # =================================================
    # 保存逐组结果
    # =================================================

    df=pd.DataFrame(
        all_results
    )


    df.to_csv(

        os.path.join(

            save_path,

            "test_metrics_per_repeat.csv"

        ),

        index=False

    )


    # =================================================
    # 汇总
    # =================================================

    metrics=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC",
        "Sensitivity",
        "Specificity"
    ]


    summary=[]


    for metric in metrics:


        values=df[metric].values


        mean=np.mean(values)


        std=np.std(
            values,
            ddof=1
        )


        ci=(
            1.96
            *
            std
            /
            np.sqrt(
                len(values)
            )
        )


        summary.append(

            {

            "Metric":
                metric,

            "Mean":
                mean,

            "Std":
                std,

            "CI95_Lower":
                max(0, mean-ci),

            "CI95_Upper":
                min(1, mean+ci)

            }

        )


    summary_df=pd.DataFrame(
        summary
    )


    summary_df.to_csv(

        os.path.join(

            save_path,

            "test_metrics_summary.csv"

        ),

        index=False

    )


    # config

    config={

        "data_path":
            data_path,

        "use_unknown":
            use_unknown,

        "num_cluster":
            num_cluster,

        "epochs":
            epochs,

        "batch_size":
            batch_size,

        "experiment_repeats":
            experiment_repeats,

        "seed":
            seed

    }


    with open(

        os.path.join(
            save_path,
            "config.json"
        ),

        "w"

    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )


    print(
        "\nFinished"
    )


    print(
        summary_df
    )


    return summary_df