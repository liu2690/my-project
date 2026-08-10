"""
模型训练模块

负责:
1. train训练
2. val验证
3. 保存最佳模型

"""

import copy

import torch
import torch.nn as nn



def train_model(
        model,
        train_loader,
        val_loader,
        epochs=30,
        lr=5e-4,
        weight_decay=5e-3,
        device=None
):

    """
    训练MultiView模型

    参数:
        epochs:
            默认30

    返回:
        best_model_state
        best_mask
        best_val_acc

    """


    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else
            "cpu"
        )


    model.to(device)


    optimizer = torch.optim.Adam(
        [
            {
                "params":
                model.logist.parameters(),
                "lr":1e-5,
                "weight_decay":1e-4
            },

            {
                "params":
                model.mlp_classifier.parameters(),
                "lr":lr,
                "weight_decay":weight_decay
            }

        ]
    )


    criterion = nn.CrossEntropyLoss()


    best_val_acc = 0.0

    best_state = None

    best_mask = None



    for epoch in range(epochs):


        # =========================
        # temperature退火
        # =========================

        if hasattr(
            model,
            "temperature"
        ):

            model.temperature = (
                1.0
                -
                0.7 *
                epoch /
                max(1,epochs-1)
            )


        # =========================
        # train
        # =========================

        model.train()


        for x,y in train_loader:


            x=x.to(
                device
            )

            y=y.to(
                device
            )


            optimizer.zero_grad()


            output,_ = model(x)


            loss = criterion(
                output,
                y
            )


            loss.backward()


            optimizer.step()



        # =========================
        # validation
        # =========================

        model.eval()


        correct=0

        total=0


        with torch.no_grad():

            for x,y in val_loader:

                x=x.to(device)

                y=y.to(device)


                output,_ = model(x)


                pred=torch.argmax(
                    output,
                    dim=1
                )


                correct += (
                    pred==y
                ).sum().item()


                total += len(y)



        val_acc = (
            correct/total
            if total>0
            else 0
        )


        # 保存最佳模型

        if val_acc > best_val_acc:


            best_val_acc = val_acc


            best_state = copy.deepcopy(
                model.state_dict()
            )


            with torch.no_grad():

                x_val = torch.cat(
                    [
                        x.to(device)
                        for x,_ in val_loader
                    ]
                )


                best_mask = (
                    model.get_score(
                        x_val
                    )
                    .detach()
                    .cpu()
                )



        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Val Acc={val_acc:.4f}"
        )


    return (
        best_state,
        best_mask,
        best_val_acc
    )