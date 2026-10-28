# EGNet 编码器来源与 CPU 核验

## 找到的官方资源

- [EGNet 官方代码](https://github.com/JXingZhao/EGNet)，ICCV 2019，Edge Guidance Network for Salient Object Detection。
- [官方论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_EGNet_Edge_Guidance_Network_for_Salient_Object_Detection_ICCV_2019_paper.pdf)，3.2 节说明额外 Conv6 分支，4.1 节说明在 DUTS 上训练。
- [官方 README 指向的模型包](https://drive.google.com/open?id=17Ffc6V5EiujtcFKupsJXhtlQ3cLK5OGp)，下载文件为 model.zip，818,669,102 字节。

实际下载并检查得到：

| 文件 | 大小（字节） | 内容 |
|---|---:|---|
| epoch_vgg.pth | 432,297,722 | 112 项张量的 EGNet VGG16 完整网络 state_dict，含 Conv6 和显著性/边缘解码器 |
| epoch_resnet.pth | 447,062,559 | 406 项 state_dict，ResNet 分支，不能当 VGG16 加载 |

SHA256：

```text
model.zip
73f8c290819b1fab852c49736244c8d994b7b89e93b09eb7ddf71762fcb79faf
epoch_vgg.pth
dbca677f39d3042ad7481fe10fe5141284ac82bfe6ec769bb74c379f1c6a6f6b
epoch_resnet.pth
38e000887ec6445f91faac7dac9539daae33ab05eb98ce6c80ba82fff0f836b7
```

模型文件不放进 Git；从官方链接下载。这里核验的是显著性训练后的 `epoch_vgg.pth`，不是 README 中 initial model 链接提供的初始化权重。

## 与原仓库的对应证据

原 GNN_SOD 与 EGNet 官方代码共享以下接口线索：

1. 原预处理默认模型文件名为 `epoch_resnet.pth`，与 EGNet `run.py` 完全相同。
2. 缺失的 `Solver(train_loader, test_loader, config, save_fold=None)` 构造形式与 EGNet `solver.py` 一致。
3. 原代码使用 BGR 输入和 `(104.00699, 116.66877, 122.67892)` 像素均值，与 EGNet `dataset.py` 一致。
4. EGNet 的 VGG16 附加三层 512 通道、3×3、无偏置卷积，形成输入 256 时的 8×8×512 F6，符合文稿图 1 的六个 Block 尺寸；标准 VGG19 的 pool5 本身没有这三层卷积。

这些证据支持“原接口很可能沿用了 EGNet”的推断，**不能证明作者实际使用了官方的这份 checkpoint**。原仓库调用 `Solver.test(imgs=...)` 返回特征，而官方 Solver 的 `test(test_mode=0)` 遍历测试集、导出显著图，不是同一个方法接口。因此直接拷入原版 Solver 仍不能运行，需要显式的特征适配器。

## 已验证的特征接口

`encoder.py` 新增 `EGNetVGG16Encoder`，可选参数为 `--encoder egnet-vgg16`。只读取官方 checkpoint 中的主干和 Conv6 权重，分别严格检查键名/尺寸；本图网络使用自己的解码器，不加载 EGNet 的 merge1/merge2 解码器。使用官方的 3×3、stride=2、padding=1 最大池化及 BGR 像素均值归一化。

在 CPU、输入 `[1,3,256,256]` 下：

| 特征 | 输出尺寸 BCHW | 在 GNN_SOD 中的用途 |
|---|---|---|
| F2，relu2_2 | 1×128×128×128 | 第四级 CNN 融合 |
| F3，relu3_3 | 1×256×64×64 | 第三级 CNN 融合 |
| F4，relu4_3 | 1×512×32×32 | 图节点与第二级 CNN 融合 |
| F5，relu5_3 | 1×512×16×16 | 图节点与第一级 CNN 融合 |
| F6，额外三层卷积 | 1×512×8×8 | 图节点 |

因此图节点取 F4/F5/F6，通道为 512+512+512=1536；四级 CNN 融合取 F5/F4/F3/F2，通道为 512/512/256/128。

测试用同一份下载权重分别加载 EGNet 官方 `vgg.py` 与本适配器，输入相同的张量，**七个返回特征与官方对应层逐元素完全一致（rtol=0、atol=0）**。同时测试了本地编码器 checkpoint 导出/加载、权重摘要一致和错误分支权重拒绝。测试代码不自动下载或执行外部代码，须显式给出已检查的官方文件路径：

```bash
GNN_SOD_EGNET_CHECKPOINT=/path/to/epoch_vgg.pth \
GNN_SOD_EGNET_REFERENCE=/path/to/official/vgg.py \
python -m pytest -q
```

本次包含实际官方权重的完整测试结果：**11 passed in 9.37s**。未提供这些路径时，官方参考对比测试跳过，其余 10 项离线测试仍可运行。

另外，已对两张合成图做 256×256 真实 SNIC 与 EGNet 特征缓存，使用修正为逐级融合的图解码器完成 batch=2 的一次 Adam 更新，五头总损失为 13.032805，数值有限。随后通过独立的无标签图像完成 checkpoint 加载和预测，输出为原始 86×66 灰度 PNG。该损失仅说明流程可执行，不是模型质量指标。

## 尚不能确认的内容

- 文稿反复写 VGG19；EGNet 官方可用分支为 VGG16 和 ResNet。采用 EGNet VGG16 需要核实并统一文稿中的主干名称，不能自动把 VGG16 当成 VGG19。
- 官方 EGNet 权重在 DUTS 上训练；文稿写在 EORSSD 训练集上微调。这是不同训练来源。本次没有进行 EORSSD 微调。
- 缺失 Solver 的特征返回细节和原实验 checkpoint 未找回。结构尺寸匹配不等于恢复了原实验模型。
- 本次的图网络训练样本是合成图，没有全量 EORSSD 指标或跨数据集结论。

因此当前保持默认 VGG19 可运行分支，同时提供已经实际核验的 EGNet VGG16 可选分支，等待原训练记录或原权重解决文稿与实现的身份问题。
