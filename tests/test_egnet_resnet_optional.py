"""Optional parity test using explicitly supplied official EGNet files.

Set GNN_SOD_EGNET_RESNET_CHECKPOINT and GNN_SOD_EGNET_RESNET_REFERENCE_DIR.
The reference directory must contain checked official resnet.py/model.py/vgg.py.
"""
import importlib.util
import os
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch
from encoder import EGNetResNet50Encoder
from runtime import configure


def load_reference_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not (os.environ.get("GNN_SOD_EGNET_RESNET_CHECKPOINT") and
                        os.environ.get("GNN_SOD_EGNET_RESNET_REFERENCE_DIR")),
                    reason="Official EGNet ResNet weights/reference paths were not supplied")
def test_resnet_official_features_conversion_and_checkpoint_roundtrip(tmp_path):
    configure("cpu", threads=2, seed=0)
    weights = os.environ["GNN_SOD_EGNET_RESNET_CHECKPOINT"]
    root = Path(os.environ["GNN_SOD_EGNET_RESNET_REFERENCE_DIR"])
    official_resnet = load_reference_module("official_egnet_resnet", root / "resnet.py")
    official_vgg = load_reference_module("official_egnet_vgg", root / "vgg.py")
    with patch.dict(sys.modules, {"resnet": official_resnet, "vgg": official_vgg}):
        official_model = load_reference_module("official_egnet_model", root / "model.py")
    reference = official_model.build_model("resnet").eval()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    reference.load_state_dict(checkpoint, strict=True)
    encoder = EGNetResNet50Encoder(weights)
    for batch, height, width in ((1, 256, 256), (2, 255, 257)):
        rgb = torch.rand(batch, 3, height, width)
        with torch.no_grad():
            expected = reference.convert(reference.base(rgb[:, [2, 1, 0]] * 255 - encoder.mean_bgr))
            high, skips = encoder(rgb)
        if height == 256:
            assert [tuple(f.shape) for f in expected] == [
                (1, 128, 128, 128), (1, 256, 65, 65), (1, 512, 33, 33),
                (1, 512, 17, 17), (1, 512, 17, 17)]
        for actual, target in zip(high + skips, expected[2:] + expected[3::-1]):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
            assert torch.isfinite(actual).all()
    assert all(not p.requires_grad for p in encoder.parameters())
    assert not any(m.training for m in encoder.modules())
    torch.save({"encoder_state": encoder.state_dict()}, tmp_path / "encoder.pt")
    restored = EGNetResNet50Encoder(tmp_path / "encoder.pt")
    assert restored.fingerprint() == encoder.fingerprint()
    del restored
    # A partial projection checkpoint must not silently leave random channels.
    incomplete = {k: v for k, v in checkpoint.items()
                  if k.startswith(("base.", "convert.")) and k != "convert.convert0.4.0.weight"}
    torch.save(incomplete, tmp_path / "missing_projection.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        EGNetResNet50Encoder(tmp_path / "missing_projection.pt")
    torch.save({"base.base.0.weight": torch.zeros(64, 3, 3, 3)}, tmp_path / "vgg.pt")
    with pytest.raises(ValueError, match="epoch_resnet.pth"):
        EGNetResNet50Encoder(tmp_path / "vgg.pt")
