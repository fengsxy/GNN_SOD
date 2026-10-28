"""Build portable, versioned SNIC/VGG19 caches from an explicit image manifest."""
import argparse
import json
from pathlib import Path
import shlex
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from skimage.color import rgb2lab
from pysnic.algorithms.snic import snic
import torch
from torch.nn import functional as F
from encoder import FeatureEncoder
from runtime import configure


def read_manifest(path):
    path = Path(path).resolve()
    samples = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = shlex.split(line, comments=True)
        if not fields:
            continue
        if len(fields) not in (1, 2):
            raise ValueError(f"{path}:{lineno}: expected image_path [mask_path]; quote paths with spaces")
        files = [(path.parent / name).resolve() for name in fields]
        for file in files:
            if not file.is_file():
                raise FileNotFoundError(f"{path}:{lineno}: {file}")
        samples.append((files[0], files[1] if len(files) == 2 else None))
    if not samples:
        raise ValueError("The image manifest is empty")
    return samples


def pool_regions(feature, labels):
    """Mean-pool [1,C,H,W] features to [N,C] without severing gradients."""
    labels = labels.to(feature.device).reshape(-1)
    pixels = feature[0].permute(1, 2, 0).reshape(-1, feature.shape[1])
    count = torch.bincount(labels)
    pooled = pixels.new_zeros((count.numel(), pixels.shape[1]))
    pooled.index_add_(0, labels, pixels)
    return pooled / count[:, None]


def spatial_graph(coord, knn):
    """Preserve the released two-pass column/row KNN sparsification.

    coord is normalized (row/height, col/width). This is not guaranteed to be
    symmetric or to retain exactly k incoming edges after the second pass.
    """
    weights = np.exp(-cdist(coord, coord) / (0.1 * np.pi) ** 2)
    np.fill_diagonal(weights, 0)
    if knn < weights.shape[0]:
        np.put_along_axis(weights, np.argsort(weights, axis=0)[:-knn, :], 0, axis=0)
        np.put_along_axis(weights, np.argsort(weights, axis=1)[:, :-knn], 0, axis=1)
    source, target = np.nonzero(weights)
    return (torch.tensor(np.stack((source, target)), dtype=torch.long),
            torch.tensor(weights[source, target], dtype=torch.float32))


def preprocess(args):
    device = configure(args.device, args.threads, args.seed)
    if args.size < 32 or args.size % 32:
        raise ValueError("--size must be a multiple of 32, at least 32")
    if args.knn < 1 or not args.segments or any(n < 2 or n > args.size ** 2 for n in args.segments):
        raise ValueError("--knn must be positive; segment counts must be in [2, size*size]")
    samples = read_manifest(args.manifest)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Cache directory must be empty to avoid mixing datasets: {out}")
    encoder = FeatureEncoder(args.weights, args.seed, args.encoder_checkpoint).to(device)
    config = {"schema_version": 1, "size": args.size, "segments": args.segments, "knn": args.knn,
              "encoder": "vgg19-pool5", "feature_signature": encoder.fingerprint(),
              "weights": "checkpoint" if args.encoder_checkpoint else args.weights,
              "seed": args.seed, "node_channels": 1536, "node_activation": "sigmoid",
              "topology": "legacy-two-pass-knn-v1",
              "preprocess": "rgb-imagenet-bilinear-mask-nearest-v1"}
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (image_path, mask_path) in enumerate(samples):
        with Image.open(image_path) as opened:
            original_size = [opened.height, opened.width]
            rgb = np.asarray(opened.convert("RGB").resize((args.size, args.size),
                             Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
        high, cnn = encoder(tensor)
        target = None
        if mask_path:
            with Image.open(mask_path) as mask:
                if [mask.height, mask.width] != original_size:
                    raise ValueError(f"Image and mask dimensions differ: {image_path}, {mask_path}")
                array = np.asarray(mask.convert("L").resize((args.size, args.size),
                                   Image.Resampling.NEAREST))
            # Accept conventional 0/255 masks as well as binary 0/1 masks.
            threshold = 0 if array.max() <= 1 else 127
            target = torch.from_numpy((array > threshold).astype(np.float32)).unsqueeze(0)
        lab = rgb2lab(rgb).tolist()
        yy, xx = torch.meshgrid(torch.arange(args.size), torch.arange(args.size), indexing="ij")
        coordinates = torch.stack((yy, xx)).float().unsqueeze(0) / args.size
        graphs = []
        for segments in args.segments:
            segmentation, _, _ = snic(lab, segments, 1)
            _, labels = np.unique(np.asarray(segmentation), return_inverse=True)
            labels = torch.from_numpy(labels.reshape(args.size, args.size)).long()
            if labels.max() < 1:
                raise ValueError("SNIC produced fewer than two regions; increase --segments")
            # Each high-level layer has 512 channels: F4 + F5 + F6 -> 1536.
            nodes = torch.cat([pool_regions(torch.sigmoid(F.interpolate(
                feature, size=(args.size, args.size), mode="bilinear", align_corners=False)), labels)
                for feature in high], dim=1).cpu()
            centers = pool_regions(coordinates, labels).numpy()
            edge_index, edge_attr = spatial_graph(centers, args.knn)
            graphs.append({"x": nodes, "edge_index": edge_index, "edge_attr": edge_attr,
                           "sp": labels.unsqueeze(0)})
        sample_id = f"{index:06d}"
        entry = {"file": sample_id + ".pt", "sample_id": sample_id,
                 "image_path": str(image_path), "mask_path": str(mask_path) if mask_path else None}
        sample = {"schema_version": 1, "config": config, "graphs": graphs,
                  "cnn": [f.squeeze(0).cpu() for f in cnn], "target": target,
                  "original_size": original_size, **entry}
        torch.save(sample, out / entry["file"])
        entries.append(entry)
        print(f"[{index + 1}/{len(samples)}] {image_path.name}: nodes={[g['x'].shape[0] for g in graphs]}",
              flush=True)
    (out / "manifest.json").write_text(json.dumps({"config": config, "samples": entries},
                                                indent=2), encoding="utf-8")
    return config


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--segments", type=int, nargs="+", default=[50, 100])
    p.add_argument("--knn", type=int, default=32)
    p.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    p.add_argument("--encoder-checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    preprocess(parser().parse_args())
