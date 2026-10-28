"""Run against downloaded official weights/code when explicitly provided.

GNN_SOD_EGNET_CHECKPOINT=/path/epoch_vgg.pth \
GNN_SOD_EGNET_REFERENCE=/path/official/vgg.py python -m pytest -q
No downloads or external-code imports occur unless both paths are supplied.
"""
import importlib.util
import os
import pytest
import torch
from encoder import EGNetVGG16Encoder
from runtime import configure


@pytest.mark.skipif(not (os.environ.get("GNN_SOD_EGNET_CHECKPOINT") and
                        os.environ.get("GNN_SOD_EGNET_REFERENCE")),
                    reason="Official EGNet weights/reference paths were not supplied")
def test_official_weights_features_and_encoder_roundtrip(tmp_path):
    configure("cpu", threads=2, seed=0)
    weights = os.environ["GNN_SOD_EGNET_CHECKPOINT"]
    spec = importlib.util.spec_from_file_location("official_egnet_vgg",
                                                 os.environ["GNN_SOD_EGNET_REFERENCE"])
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    reference = official.vgg16().eval()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    reference.load_state_dict({k.removeprefix("base."): v for k, v in checkpoint.items()
                               if k.startswith("base.")}, strict=True)
    encoder = EGNetVGG16Encoder(weights)
    rgb = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        expected = reference(rgb[:, [2, 1, 0]] * 255 - encoder.mean_bgr)
        high, skips = encoder(rgb)
    assert [tuple(f.shape) for f in expected] == [
        (1, 128, 128, 128), (1, 256, 64, 64), (1, 512, 32, 32),
        (1, 512, 16, 16), (1, 512, 8, 8)]
    for actual, target in zip(high + skips, expected[2:] + expected[3::-1]):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    torch.save({"encoder_state": encoder.state_dict()}, tmp_path / "encoder.pt")
    restored = EGNetVGG16Encoder(tmp_path / "encoder.pt")
    assert restored.fingerprint() == encoder.fingerprint()
    torch.save({"base.conv1.weight": torch.zeros(64, 3, 7, 7)}, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="epoch_vgg.pth"):
        EGNetVGG16Encoder(tmp_path / "wrong.pt")
