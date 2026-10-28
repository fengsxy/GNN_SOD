# GNN_SOD

基于多尺度图卷积的光学遥感图像显著性检测。模型先提取 CNN 特征，通过 SNIC 超像素分割构建多尺度图，再融合图卷积特征与 CNN 特征，输出显著性图。

项目提供预处理、训练、断点续训和预测脚本，支持 VGG19、EGNet VGG16 和 EGNet ResNet50 三种编码器，可选择 CPU 或 NVIDIA GPU。

## 安装

环境配置：

| 组件 | CPU | NVIDIA GPU |
|---|---|---|
| Python | 3.12 | 3.12 |
| PyTorch | 2.5.1+cpu | 2.5.1+cu124 |
| torchvision | 0.20.1+cpu | 0.20.1+cu124 |
| PyTorch Geometric | 2.6.1 | 2.6.1 |
| CUDA 运行时 | — | 12.4 |

CPU 和 GPU 分别使用 [requirements-cpu.txt](requirements-cpu.txt) 与 [requirements-gpu.txt](requirements-gpu.txt)，两者都包含 [公共依赖](requirements.txt)。测试依赖在 [requirements-dev.txt](requirements-dev.txt)。PyTorch 版本组合参考[官方安装说明](https://pytorch.org/get-started/previous-versions/#v251)。

以下为 Linux x86_64 安装命令。在项目目录中创建 Python 3.12 环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

然后按设备选择一种安装方式。

CPU：

```bash
python -m pip install -r requirements-cpu.txt
```

NVIDIA GPU（需要支持 CUDA 12.4 的驱动，先确认 `nvidia-smi` 能识别显卡）：

```bash
python -m pip install -r requirements-gpu.txt
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
```

GPU 环境最后一项应为 `True`。本项目使用的 PyG 运算无需另外安装 torch-scatter 或 torch-sparse。

## 快速体验

```bash
python smoke_test.py --work-dir runs/demo
```

在 GPU 上运行：

```bash
python smoke_test.py --work-dir runs/demo_gpu --device cuda --size 256
```

脚本会生成三张合成示例图，完成预处理、训练、续训和预测，无需准备数据或下载权重。预测图保存在 `runs/demo/predictions/`，运行结果保存在 `runs/demo/smoke_result.json`。

输出位置随 `--work-dir` 改变，运行结果也会记录设备和 PyTorch/CUDA 版本。默认使用 64×64 输入以缩短运行时间，可加 `--size 256` 使用完整输入尺寸。重复运行时，换一个新的 `--work-dir`。

所有运行脚本都接受 `--device`，默认是 `cpu`。GPU 用 `--device cuda`，多卡机器可用 `--device cuda:1` 指定一张卡。下面的数据处理、训练和预测示例使用 CPU；在每条命令后加上 GPU 参数即可切换。SNIC 分割和 KNN 构图仍在 CPU 上执行，CNN 特征提取与图网络计算使用所选设备。

## 选择编码器

| 编码器 | `--encoder` | 预训练权重 |
|---|---|---|
| VGG19（默认） | `vgg19` | ImageNet 权重，首次运行自动下载 |
| EGNet VGG16＋Conv6 | `egnet-vgg16` | `epoch_vgg.pth`，在 DUTS 上训练 |
| EGNet ResNet50＋通道转换 | `egnet-resnet50` | `epoch_resnet.pth`，在 DUTS 上训练 |

使用 EGNet 时，下载 [官方模型包](https://drive.google.com/open?id=17Ffc6V5EiujtcFKupsJXhtlQ3cLK5OGp)，将解压得到的两个 `.pth` 文件放到 `weights/` 目录。用以下命令体验 ResNet 分支：

```bash
python smoke_test.py --work-dir runs/resnet_demo --size 256 \
  --encoder egnet-resnet50 --encoder-checkpoint weights/epoch_resnet.pth
```

体验 VGG16 时，将编码器改为 `egnet-vgg16`，权重改为 `weights/epoch_vgg.pth`，并使用新的工作目录。两种 EGNet 编码器也通过 `--device cuda` 使用 GPU。

## 准备数据

将 EORSSD 或其他带显著性掩码的数据集整理成图像和掩码，并分别建立 `data/train.txt`、`data/val.txt`、`data/test.txt`。

训练和验证清单每行包含一张图像及其掩码：

```text
images/train_0001.jpg masks/train_0001.png
images/train_0002.jpg masks/train_0002.png
```

预测清单可以只写图像路径：

```text
images/test_0001.jpg
images/test_0002.jpg
```

路径相对于清单所在目录解析，例如第一行图像位于 `data/images/train_0001.jpg`；也可以使用绝对路径。带空格的路径用双引号包围。掩码使用单通道 0/255 或 0/1 二值图，尺寸与对应图像一致。训练和验证使用不同的图像。

## 预处理

预处理会提取并缓存 CNN 特征、超像素标签和图结构，后续训练直接读取缓存。

```bash
python pre_process.py --manifest data/train.txt --out-dir runs/train_cache
python pre_process.py --manifest data/val.txt --out-dir runs/val_cache
python pre_process.py --manifest data/test.txt --out-dir runs/test_cache
```

以上命令默认使用 VGG19。使用 EGNet 时，在每条预处理命令末尾添加对应参数：

| 编码器 | 添加的参数 |
|---|---|
| VGG16 | `--encoder egnet-vgg16 --encoder-checkpoint weights/epoch_vgg.pth` |
| ResNet50 | `--encoder egnet-resnet50 --encoder-checkpoint weights/epoch_resnet.pth` |

默认输入尺寸为 256×256，超像素目标数量为 50/100，K 为 32。可通过 `--size`、`--segments`、`--knn` 调整；程序会打印 SNIC 实际生成的区域数量。

同一次实验的训练、验证和预测使用相同的编码器、权重和预处理参数。切换编码器或参数后，使用新的缓存目录重新预处理，并重新训练图网络。

## 训练与续训

```bash
python Main.py --data runs/train_cache --val-data runs/val_cache \
  --out-dir runs/model --epochs 20 --batch-size 2
```

训练使用 Adam，默认学习率为 0.005。上例将 batch size 设为 2，适合先在 CPU 上运行；可根据内存或显存调整 `--batch-size`，通过 `--threads` 调整 CPU 线程数。例如，使用 GPU、batch size 为 4：

```bash
python Main.py --data runs/train_cache --val-data runs/val_cache \
  --out-dir runs/model_gpu --epochs 20 --batch-size 4 --device cuda
```

训练目录包含：

| 文件 | 内容 |
|---|---|
| `latest.pt` | 最近一轮的模型和优化器状态，用于续训 |
| `best.pt` | 验证集 MAE 最低的模型，用于预测 |
| `history.json` | 每轮训练损失和验证 MAE |

没有验证集时，省略 `--val-data`，训练会保存 `latest.pt` 和训练历史。

从第 20 轮继续训练到第 30 轮：

```bash
python Main.py --data runs/train_cache --val-data runs/val_cache \
  --out-dir runs/model --epochs 30 --batch-size 2 --resume runs/model/latest.pt
```

`--epochs` 表示总轮数。续训会恢复优化器设置，包括检查点中的学习率。

## 预测

```bash
python predict.py --data runs/test_cache --checkpoint runs/model/best.pt \
  --out-dir runs/predictions
```

如果训练时没有使用验证集，将 `best.pt` 改为 `latest.pt`。

预测结果为单通道灰度 PNG，尺寸恢复为各自原图的大小。`predictions.json` 记录输出文件与输入图像的对应关系。可通过 `--batch-size` 设置预测批量大小。

预处理缓存和预测输出使用空目录；再次运行时指定新的 `--out-dir` 即可。

## 测试与实现文档

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试覆盖图反投影、梯度、批处理、损失和检查点读写。有可用 GPU 时，测试也会检查 CUDA 下 batch=1/2 的前向、反向和参数更新。提供官方权重及参考代码后，还可运行 EGNet 特征一致性测试；具体设置见以下文档。

- [实现说明与版本兼容](docs/CPU_REPAIR.md)
- [CPU 测试记录](docs/CPU_VALIDATION.md)
- [EGNet VGG16 权重与特征接口](docs/EGNET_CHECK.md)
- [EGNet ResNet50 权重与特征接口](docs/EGNET_RESNET_CHECK.md)
