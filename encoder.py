"""Frozen VGG19 fallback and verified EGNet VGG16 feature encoder.

In the default VGG19 branch, F6 is pool5: a reconstruction of the figure's
shape, not the unavailable original fine-tuned encoder. The optional EGNet
branch uses the official trained Conv6 layers instead.
"""
import hashlib
import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights


class FeatureEncoder(nn.Module):
    name = "vgg19-pool5"
    preprocessing = "rgb-imagenet-bilinear-mask-nearest-v1"

    def __init__(self, weights="imagenet", seed=0, checkpoint=None):
        super().__init__()
        selected = VGG19_Weights.IMAGENET1K_V1 if weights == "imagenet" and not checkpoint else None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.features = vgg19(weights=selected).features
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.load_state_dict(state.get("encoder_state", state), strict=True)
        self.requires_grad_(False)
        self.eval()

    def fingerprint(self):
        digest = hashlib.sha256(b"vgg19-pool5-rgb-imagenet-normalization-v1")
        for name, value in self.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def forward(self, image):
        x = (image - self.mean) / self.std
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in (8, 17, 26, 35, 36):
                features[i] = x
        return [features[i] for i in (26, 35, 36)], [features[i] for i in (35, 26, 17, 8)]


class EGNetVGG16Encoder(nn.Module):
    """Encoder-only adapter for the official EGNet epoch_vgg.pth checkpoint.

    Architecture/weight layout: https://github.com/JXingZhao/EGNet/blob/master/vgg.py
    This is VGG16 with 3x3/stride-2/padding-1 pooling and three extra 512-channel
    convolutions after pool5. The extra convolutions, not pool5 alone, produce F6.
    Official weights are trained on DUTS; this does not imply EORSSD fine-tuning.
    EGNet saliency/edge decoder weights are intentionally not used by GNN_SOD.
    """
    name = "egnet-vgg16-conv6"
    preprocessing = "bgr-caffe-bilinear-mask-nearest-v1"

    def __init__(self, checkpoint):
        super().__init__()
        if not checkpoint:
            raise ValueError("EGNet VGG16 requires --encoder-checkpoint pointing to epoch_vgg.pth")
        layers = []
        channels = 3
        for width, count in ((64, 2), (128, 2), (256, 3), (512, 3), (512, 3)):
            for _ in range(count):
                layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=True)])
                channels = width
            layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        self.features = nn.ModuleList(layers)
        self.extra = nn.Sequential(*[
            layer for _ in range(3)
            for layer in (nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.ReLU(inplace=True))])
        self.register_buffer("mean_bgr", torch.tensor([104.00699, 116.66877, 122.67892]).view(1, 3, 1, 1))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "encoder_state" in state:
            self.load_state_dict(state["encoder_state"], strict=True)
        else:
            base = {k.removeprefix("base.base."): v for k, v in state.items()
                    if k.startswith("base.base.")}
            extra = {k.removeprefix("base.base_ex.ex."): v for k, v in state.items()
                     if k.startswith("base.base_ex.ex.")}
            if not base or not extra:
                raise ValueError("Expected official EGNet epoch_vgg.pth, including trained conv6; ResNet/ImageNet-only weights are incompatible")
            self.features.load_state_dict(base, strict=True)
            self.extra.load_state_dict(extra, strict=True)
        self.requires_grad_(False)
        self.eval()

    def fingerprint(self):
        digest = hashlib.sha256(b"egnet-vgg16-conv6-bgr-caffe-v1")
        for name, value in self.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def forward(self, image):
        x = image[:, [2, 1, 0]] * 255 - self.mean_bgr
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in (8, 15, 22, 29):
                features[i] = x
        f6 = self.extra(x)
        return [features[22], features[29], f6], [features[i] for i in (29, 22, 15, 8)]
