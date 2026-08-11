"""
测试评价模块

只使用test数据集

"""

import torch
import numpy as np

from src.metrics import compute_metrics



def evaluate_model(
        model,
        test_loader,
        device=None
):


    if device is None:

        device=torch.device(
            "cuda"
            if torch.cuda.is_available()
            else
            "cpu"
        )


    model.to(device)

    model.eval()


    preds=[]

    probs=[]

    targets=[]



    with torch.no_grad():


        for x,y in test_loader:


            x=x.to(device)


            output,_ = model(x)


            probability = torch.softmax(
                output,
                dim=1
            )[:,1]


            prediction=torch.argmax(
                output,
                dim=1
            )



            preds.extend(
                prediction.cpu()
                .numpy()
            )


            probs.extend(
                probability.cpu()
                .numpy()
            )


            targets.extend(
                y.numpy()
            )



    result = compute_metrics(
    targets,
    preds,
    probs
    )


    # 保存原始预测信息
    result["Targets"] = np.array(targets)

    result["Predictions"] = np.array(preds)

    result["Probabilities"] = np.array(probs)


    return result