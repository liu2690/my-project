"""
实验总入口

功能:
1. 加载VOC mat数据
2. train/val/test划分
3. K-means VOC聚类
4. MultiView训练
5. Test最终评价
6. 保存结果


"""

import os
import json
import random

import numpy as np

import torch
import scipy.io as sio


from src.split import split_dataset

from src.clustering import feature_cluster

from src.model import MultiView

from src.trainer import train_model

from src.evaluator import evaluate_model



# ============================================================
# 随机种子
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False



# ============================================================
# 主实验函数
# ============================================================

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


    set_seed(seed)


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )


    os.makedirs(
        save_path,
        exist_ok=True
    )


    print("="*60)

    print("Start Experiment")

    print("="*60)


    print(
        "device:",
        device
    )



    # ========================================================
    # 1. 加载mat
    # ========================================================
    if data_path is None:

        if use_unknown:

            data_path = (
                "data/"
                "voc_dataset_1+2_vs_3_with_unknown.mat"
            )

        else:

            data_path = (
                "data/"
                "voc_dataset_1+2_vs_3.mat"
            )

    data = sio.loadmat(
        data_path
    )


    X = torch.tensor(
        data["X"],
        dtype=torch.float32
    )


    y = torch.tensor(
        data["y"].reshape(-1),
        dtype=torch.long
    )


    feat_names = [

        str(x[0])

        for x in data["feat_names"].flatten()

    ]



    print(
        "samples:",
        X.shape[0]
    )

    print(
        "features:",
        X.shape[1]
    )


    print(
        "Unknown included:",
        use_unknown
    )



    # ========================================================
    # 2. train val test
    # ========================================================


    (
        train_loader,
        val_loader,
        test_loader,
        x_train

    ) = split_dataset(

        X,

        y,

        batch_size=batch_size,

        seed=seed

    )


    print(
        "train:",
        len(train_loader.dataset)
    )


    print(
        "val:",
        len(val_loader.dataset)
    )


    print(
        "test:",
        len(test_loader.dataset)
    )



    # ========================================================
    # 3. K-means聚类
    # ========================================================


    cluster_mask = feature_cluster(

        x_train.numpy(),

        num_cluster=num_cluster,

        seed=seed

    )


    print(
        "cluster mask:",
        cluster_mask.shape
    )



    # ========================================================
    # 4. 创建模型
    # ========================================================


    model = MultiView(

        cluster_mask=cluster_mask,

        num_blocks=8,

        head_dim=128,

        num_classes=2,

        temperature=1.0

    )



    # ========================================================
    # 5. train + validation
    # ========================================================


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


    print(
        "Best val accuracy:",
        best_val_acc
    )



    # ========================================================
    # 6. 保存最佳模型
    # ========================================================


    model_path = os.path.join(

        save_path,

        "best_model.pt"

    )


    torch.save(

        {

            "state_dict":
                best_state,


            "mask":
                best_mask,


            "cluster_mask":
                cluster_mask,


            "val_acc":
                best_val_acc,


            "num_cluster":
                num_cluster


        },

        model_path

    )


    print(
        "saved:",
        model_path
    )



    # ========================================================
    # 7. Test评价
    # ========================================================


    model.load_state_dict(
        best_state
    )


    test_result = evaluate_model(

        model,

        test_loader,

        device=device

    )


    print("\nTEST RESULT")

    print("-"*40)


    for key,value in test_result.items():

        if key!="Confusion_Matrix":

            print(
                f"{key}: {value:.4f}"
            )


    print(
        "Confusion Matrix:"
    )

    print(
        test_result[
            "Confusion_Matrix"
        ]
    )



    # ========================================================
    # 8. 保存结果
    # ========================================================


    result_save = {}

    for k,v in test_result.items():

        if k=="Confusion_Matrix":

            result_save[k]=v.tolist()

        else:

            result_save[k]=float(v)



    with open(

        os.path.join(
            save_path,
            "test_metrics.json"
        ),

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            result_save,

            f,

            indent=2,

            ensure_ascii=False

        )



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



    print("="*60)

    print("Experiment finished")

    print("="*60)



    return test_result