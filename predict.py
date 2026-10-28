"""Export final saliency maps as grayscale PNGs at each source image's size."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from Dataset import MyDataset
from Model import GCN
from runtime import check_config, check_model_version, configure


@torch.no_grad()
def predict(args):
    device = configure(args.device, args.threads)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    dataset = MyDataset(args.data, require_target=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    check_model_version(checkpoint)
    check_config(checkpoint["config"], dataset.config)
    model = GCN(dataset.config["node_channels"], len(dataset.config["segments"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Prediction output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    entries = {entry["sample_id"]: entry for entry in dataset.samples}
    predictions = []
    for graphs in DataLoader(dataset, batch_size=args.batch_size, num_workers=0):
        graphs = [graph.to(device) for graph in graphs]
        output = model(graphs)[-1]
        if not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite prediction")
        for i, sample_id in enumerate(graphs[0].sample_id):
            size = graphs[0].original_size[i].tolist()
            saliency = F.interpolate(output[i:i + 1], size=size, mode="bilinear", align_corners=False)
            pixels = (saliency[0, 0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            filename = sample_id + ".png"
            Image.fromarray(pixels).save(out / filename)
            predictions.append({"prediction": filename, "image_path": entries[sample_id]["image_path"],
                                "original_size": size})
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"Exported {len(predictions)} saliency maps to {out}", flush=True)
    return predictions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    return p


if __name__ == "__main__":
    predict(parser().parse_args())
