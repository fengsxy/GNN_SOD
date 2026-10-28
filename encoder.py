"""Explicit VGG19 replacement for the missing Solver/encoder checkpoint.

F6 is pool5 (512 x H/32 x W/32), a reconstruction of the figure's shape,
not a claim to recover the unavailable original fine-tuned encoder.
"""
import hashlib
import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights


class FeatureEncoder(nn.Module):
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
