# EGNet ResNet50 权重与 CPU 验证

使用已下载的官方 `epoch_resnet.pth`，447,062,559 字节，SHA256 为 `38e000887ec6445f91faac7dac9539daae33ab05eb98ce6c80ba82fff0f836b7`。下载链接及来源见 [EGNet 核验记录](EGNET_CHECK.md)。原始 GNN_SOD 预处理默认的 checkpoint 文件名正是 epoch_resnet.pth，但同名不能证明原作者实际用了这份未再微调的公开文件。

## 严格保留官方结构

对照 [EGNet resnet.py](https://github.com/JXingZhao/EGNet/blob/master/resnet.py) 和 [model.py](https://github.com/JXingZhao/EGNet/blob/master/model.py)，在 torchvision ResNet50 上恢复 EGNet 使用的结构，再严格加载 `base.*` 和 `convert.convert0.*` 权重：

- Bottleneck 的下采样 stride 位于第一个 1×1 卷积，3×3 卷积 stride 为 1。
- 3×3、stride=2、padding=1 的最大池化启用 ceil_mode。
- layer4 不再下采样，全部 3×3 卷积 dilation=2、padding=2。
- 五个训练过的 1×1+ReLU 转换层分别完成 64→128、256→256、512→512、1024→512、2048→512。
- BGR 输入减 `(104.00699, 116.66877, 122.67892)`，主干和投影均冻结，BatchNorm 处于 eval 模式。

适配器名称为 `EGNetResNet50Encoder`，命令行参数为 `--encoder egnet-resnet50`。未使用官方 EGNet 的 merge1/merge2 解码器，没有用新随机投影代替已训练的 ConvertLayer。新适配器保留独立的 encoder 标识和权重摘要，防止与其他编码器缓存混用。

## 官方参考对比

完整官方 EGNet ResNet 网络严格加载下载权重成功（406 项 state_dict 全部匹配）。从其 `base` 和 `convert` 模块取得参考特征，与新适配器的输出比较。对于 256×256 输入：

| 层级 | 原始通道 | 转换后通道 | 空间尺寸 | 图网络用途 |
|---|---:|---:|---|---|
| stem | 64 | 128 | 128×128 | 第四级 CNN 融合 |
| layer1 | 256 | 256 | 65×65 | 第三级 CNN 融合 |
| layer2 | 512 | 512 | 33×33 | 图节点、第二级 CNN 融合 |
| layer3 | 1024 | 512 | 17×17 | 图节点、第一级 CNN 融合 |
| layer4 | 2048 | 512 | 17×17 | 图节点 |

最后三级转换后特征组成 1536 维图节点输入；第 4/3/2/1 个转换后特征用于四级 CNN 融合，通道数为 512/512/256/128。各层按图网络输入图像尺寸插值，允许空间大小与 VGG 特征不同。

七个返回特征与官方实现的对应特征 **逐元素完全一致，最大绝对误差为 0（rtol=0、atol=0）**。测试同时覆盖 batch=1、256×256 和 batch=2、255×257；另验证编码器 checkpoint 导出/恢复、冻结/eval 状态、VGG 权重误用拒绝，以及缺失投影权重时的严格加载失败。

```bash
GNN_SOD_EGNET_CHECKPOINT=/path/to/epoch_vgg.pth \
GNN_SOD_EGNET_REFERENCE=/path/to/EGNet/vgg.py \
GNN_SOD_EGNET_RESNET_CHECKPOINT=/path/to/epoch_resnet.pth \
GNN_SOD_EGNET_RESNET_REFERENCE_DIR=/path/to/EGNet \
python -m pytest -q
```

参考目录需要包含经过检查的官方 resnet.py、vgg.py、model.py；测试只在显式提供路径时导入这些代码，不自动下载。此次整体结果为 **12 passed in 12.91s**。

## 完整图网络流程

```bash
python smoke_test.py --work-dir runs/resnet_smoke --size 256 --encoder egnet-resnet50 --encoder-checkpoint /path/to/epoch_resnet.pth
```

本次使用实际官方权重、SNIC、PyG 和三张合成图（训练两张、独立验证一张，预测清单不含标签）：

- 目标超像素 50/100，实际 64/121；K=32。
- batch=2 完成第一轮训练，五头总损失 13.076864。
- 从 latest.pt 续训，batch=1 完成第二轮，平均损失 12.652222。
- 梯度和损失均为有限值，验证选模及 latest/best 检查点保存成功。首个图卷积 linB 权重相对同 seed 初始化的最大变化为 0.015021，确认发生实际参数更新。
- ResNet 缓存与 EGNet VGG16 缓存混用时，encoder 与 feature_signature 检查明确拒绝。
- 无标签预测导出成功，PNG 为原图尺寸 86×66 的灰度图。
- 单次完整自检用时 39.44 s；这是包括进程启动、预处理和验证的墙钟时间，并有其他并发任务，不用于报告 FPS 或比较模型速度。

这些结果证明 ResNet 特征可用于当前图网络的训练、续训和预测，不提供遥感精度结论。

## 与文稿的关系

通道接口兼容，但空间层级与文稿图 1 不同：EGNet ResNet 在 256 输入时最深层为 17×17，没有图示 VGG 路径的 8×8 第六卷积块。也不能把 ResNet50 写成 VGG19。公开权重仍来自 DUTS 显著性训练，不是本次完成的 EORSSD 微调。原文实际采用哪种主干、是否在公开权重上继续训练，仍需原训练记录或原 checkpoint 佐证。
