# BPD 四组实验复现代码

本仓库公开四组 BPD 实验的代码、固定参数和已发表结果，但**不公开原始 MAT 数据**。获得数据使用许可后，将数据按 [`data/README.md`](data/README.md) 的文件名和字段说明放入本地 `data/`，即可运行下列实验。

## 四组实验

| 编号 | 实验 | 入口 | 已归档结果 |
|---|---|---|---|
| 01 | 删除棉球物质后的 BPD-MLP | `实验归档/01_删除棉球物质_BPD_MLP_ACC78.12/run.py` | 独立测试 ACC 78.12% |
| 02 | 同数据 BPD-CNN 对照 | `实验归档/02_BPD_CNN对照_ACC65.62/run.py` | 独立测试 ACC 65.62% |
| 03 | VOC 选择频率 60/70/85% 筛选 | `实验归档/03_VOC选择频率筛选_60_70_85/run.py` | 内部验证 |
| 04 | 264 VOC 的 L2 逻辑回归 | `实验归档/04_264VOC_L2逻辑回归_ACC78.13/run.py` | 内部验证 ACC 78.13% |

## 环境

已验证环境：Python 3.10.20、PyTorch 2.7.0（CUDA 12.8）。建议使用 Python 3.10：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`。如需 GPU，请按 PyTorch 官方说明安装与本机 CUDA 匹配的 PyTorch，再安装其余依赖。

## 数据检查

四个入口都支持只检查环境、数据字段和复现配置而不开始训练：

```bash
python "实验归档/01_删除棉球物质_BPD_MLP_ACC78.12/run.py" --check
python "实验归档/02_BPD_CNN对照_ACC65.62/run.py" --check
python "实验归档/03_VOC选择频率筛选_60_70_85/run.py" --check
python "实验归档/04_264VOC_L2逻辑回归_ACC78.13/run.py" --check
```

## 完整复现

从仓库根目录依次运行：

```bash
python "实验归档/01_删除棉球物质_BPD_MLP_ACC78.12/run.py"
python "实验归档/02_BPD_CNN对照_ACC65.62/run.py"
python "实验归档/03_VOC选择频率筛选_60_70_85/run.py"
python "实验归档/04_264VOC_L2逻辑回归_ACC78.13/run.py"
```

新生成文件统一写入已被 Git 忽略的 `result/reproduced/`，不会覆盖仓库中的归档结果。01 和 02 均训练 100 个候选模型，运行时间明显长于 03 和 04。

固定随机种子、数据划分和超参数已写入代码。深度学习结果仍可能因 GPU、CUDA、PyTorch 或底层算子不同出现轻微浮点差异；仓库中的 CSV、JSON、图片和模型文件是原验证环境的归档结果。

## 数据与隐私

- `.gitignore` 排除 `*.mat`、`*.xlsx`、`result/` 和本地环境。
- 提交前建议执行 `git ls-files "*.mat" "*.xlsx"`，输出必须为空。
- 原始数据只能通过数据申请或授权渠道获取，不应提交到 Git 或公开网盘。
