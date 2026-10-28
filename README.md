# GNN_SOD：可在 CPU 运行的修复版

光学遥感图像显著性检测：VGG19 离线特征 → 多尺度 SNIC 图 → GatedEdgeConv 解码器 → 显著图。

本版本补齐原仓库缺失的预处理接口，并修复 CPU、数据缓存、通道、尺寸及批处理问题。**这是运行流程修复，不是论文精度复现。** 原始 `Solver`、在 EORSSD 上微调过的编码器权重和训练配方未提供；这里显式使用 torchvision VGG19，F6 采用 pool5。默认加载 ImageNet 权重，离线自检使用固定随机权重。完整差异见 [修复说明](docs/CPU_REPAIR.md)。

## 安装

已在 Linux、Python 3.12、CPU 上验证。建议使用独立环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

这里只使用 PyG 的基础 MessagePassing，不需要另行编译 torch-scatter/torch-sparse。

## 一条命令检查完整流程

```bash
python smoke_test.py --work-dir runs/smoke
```

该命令生成 3 张合成图，实际调用 SNIC、VGG19、训练及预测入口，覆盖：

- 2 张训练图（含空掩码）、1 张独立验证图，以及无标签预测清单；
- batch size 2 训练，然后 batch size 1 从检查点继续训练；
- 五个监督输出的 BCE + SSIM + IoU 损失、验证 MAE、检查点读写；
- 预测显著图恢复到原图尺寸，输出 PNG 和路径映射 JSON。

默认用 64×64 加快自检。验证论文输入尺寸：

```bash
python smoke_test.py --work-dir runs/smoke256 --size 256
python -m pytest -q
```

每次使用新的空工作目录。结果保存在 `runs/smoke256/smoke_result.json`。**合成图的损失和 MAE 不能用作论文实验结果。** 自检不需要下载数据集或预训练权重。

## 准备真实数据

本仓库不包含 EORSSD 数据、原始微调权重或已经训练好的显著性模型。将获得的数据组织成图像/掩码，并为每个划分建立 UTF-8 清单。例如 `data/train.txt`：

```text
images/train_0001.jpg masks/train_0001.png
images/train_0002.jpg masks/train_0002.png
"images/name with spaces.jpg" "masks/name with spaces.png"
```

路径相对于**清单所在目录**解析，也可使用绝对路径。每行一张图；训练/验证必须同时有掩码，推理只需第一列。掩码为灰度二值图，支持 0/255 和 0/1；图像与掩码的原始尺寸必须相同。空白行和 `#` 注释可用。不同划分使用不同缓存目录。

```bash
python pre_process.py --manifest data/train.txt --out-dir runs/train_cache --device cpu
python pre_process.py --manifest data/val.txt --out-dir runs/val_cache --device cpu
python pre_process.py --manifest data/test.txt --out-dir runs/test_cache --device cpu
```

默认 `--size 256 --segments 50 100 --knn 32 --weights imagenet --threads 2`。首次使用 ImageNet 权重会下载 torchvision 的 VGG19 权重（约 550 MB）。SNIC 的参数是目标区域数，实际区域数受网格播种影响，不一定恰好为 50/100；程序会打印实际数量。训练、验证和预测必须使用同一编码器和同一组预处理参数。

若有按本版本 `FeatureEncoder` 结构训练的编码器，可使用：

```bash
python pre_process.py --manifest data/train.txt --out-dir runs/train_cache --encoder-checkpoint encoder.pt
```

`encoder.pt` 应为 `FeatureEncoder.state_dict()`，或 `{"encoder_state": state_dict}`。加载采用严格键名/尺寸检查；未知结构的旧 `epoch_resnet.pth` 不会被自动转换。本版本**没有补造原编码器的微调过程**。更换编码器后需重新生成所有缓存并重新训练解码器。`--weights none --seed 0` 仅供调试，不能作为论文预训练模型。

## 训练与续训

```bash
python Main.py --data runs/train_cache --val-data runs/val_cache --out-dir runs/model --epochs 20 --batch-size 2 --device cpu
```

默认学习率 0.005、Adam、weight decay 0，继承原训练脚本；默认 batch size 为文稿中的 4，上例用 2 降低 CPU 内存需求。`--threads` 可调节 CPU 线程数。实际训练时间和内存随图像大小、批量大小、设备变化，自检耗时不代表完整 EORSSD 训练耗时。

- `latest.pt`：最近一轮的模型、优化器、随机状态、缓存配置和训练历史。
- `best.pt`：提供验证集时，按验证 MAE 选择的检查点。
- `history.json`：每轮训练损失和可选验证 MAE。这里的 MAE 在预处理后的尺寸上计算，仅用于验证选模。

不提供 `--val-data` 也能训练，但只保存 `latest.pt`，不把训练损失作为验证结果。训练和验证清单不得共享同一路径的图像；不要用测试集选模。

```bash
python Main.py --data runs/train_cache --val-data runs/val_cache --out-dir runs/model --epochs 30 --batch-size 2 --resume runs/model/latest.pt
```

`--epochs` 是训练总轮数。续训恢复检查点中的优化器设置（包括学习率）；CPU 随机状态也会恢复。

## 预测

```bash
python predict.py --data runs/test_cache --checkpoint runs/model/best.pt --out-dir runs/predictions --device cpu
```

没有验证集时，改用 `latest.pt`。输出 `000000.png` 等单通道 8-bit 显著图，尺寸与各自原图一致；`predictions.json` 记录输出文件与输入图像的对应关系。支持 `--batch-size`，输入清单可没有标签。

缓存和检查点记录编码器权重摘要、图尺度、K、图构造方式、尺寸和特征接口，不兼容时会明确报错。旧版本 `.pkl` 和模型权重不兼容此版本，需从原图重建缓存并重新训练。缓存、模型和预测输出不纳入 Git。

## 方法接口

| 用途 | VGG19 特征 | 通道/输出 |
|---|---|---|
| 图节点 | F4=relu4_4、F5=relu5_4、F6=pool5；双线性放大后 sigmoid、区域均值 | 1536 |
| 四级 CNN 融合 | F5、F4、F3=relu3_4、F2=relu2_2 | 512、512、256、128 |
| 各尺度图卷积 | 保留原 GatedEdgeConv 和 GraphSizeNorm | 64→32→16→8 |
| 多尺度融合 | 各分支相加、1×1 卷积；沿用原代码首级融合图的跨级连接 | 32、16、8、4 |
| 显著性输出 | 四个侧输出和一个融合输出 | 均为 `[B,1,H,W]` |

坐标只用于构图，不混入 1536 维 CNN 节点特征。图权重与原代码一致：归一化质心距离经 `exp(-distance / (0.1*pi)^2)` 加权，再执行原来的列/行两遍 KNN 稀疏化，去掉自环。该操作不保证每个节点最后恰有 K 条边，也不保证图对称。

评审要求的跨数据集实验、核心模块消融、参数敏感性、完整复杂度指标及近期方法比较，仍需在真实数据上另行完成；本次修复不构成这些实验结果。
