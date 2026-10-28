"""Train the graph decoder from cached VGG19/SNIC features on CPU or CUDA."""
import argparse
import json
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from Dataset import MyDataset
from Model import GCN
from losses import mixed_loss
from runtime import check_config, configure


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    error, count = 0.0, 0
    for graphs in loader:
        graphs = [graph.to(device) for graph in graphs]
        pred = model(graphs)[-1]
        error += (pred - graphs[0].y).abs().mean((1, 2, 3)).sum().item()
        count += pred.shape[0]
    return error / count


def train(args):
    device = configure(args.device, args.threads, args.seed)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    dataset = MyDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = None
    if args.val_data:
        val_data = MyDataset(args.val_data)
        check_config(dataset.config, val_data.config)
        train_paths = {s["image_path"] for s in dataset.samples}
        if train_paths.intersection(s["image_path"] for s in val_data.samples):
            raise ValueError("Training and validation manifests share images")
        val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0)
    model = GCN(dataset.config["node_channels"], len(dataset.config["segments"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    start, best, history = 0, float("inf"), []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        check_config(checkpoint["config"], dataset.config)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start, best = checkpoint["epoch"], checkpoint["best_val_mae"]
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["rng_state"])
        if start >= args.epochs:
            raise ValueError("--epochs is the total epoch count and must exceed the resumed epoch")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "latest.pt").exists() and not args.resume:
        raise ValueError("Output has a checkpoint; use --resume or a new --out-dir")
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for graphs in loader:
            graphs = [graph.to(device) for graph in graphs]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(graphs)
            loss = sum(mixed_loss(prediction, graphs[0].y) for prediction in outputs)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite gradient")
            optimizer.step()
            batch = graphs[0].num_graphs
            total += loss.item() * batch
            seen += batch
        record = {"epoch": epoch, "train_loss": total / seen, "samples": seen,
                  "steps": len(loader)}
        improved = False
        if val_loader:
            record["val_mae"] = validate(model, val_loader, device)
            improved = record["val_mae"] < best
            best = min(best, record["val_mae"])
        history.append(record)
        checkpoint = {"format_version": 1, "config": dataset.config, "epoch": epoch,
                      "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "best_val_mae": best, "history": history, "rng_state": torch.get_rng_state()}
        temporary = out / "latest.tmp.pt"
        torch.save(checkpoint, temporary)
        temporary.replace(out / "latest.pt")
        if improved:
            torch.save(checkpoint, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
    return history


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--val-data")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=.005)
    p.add_argument("--resume")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    train(parser().parse_args())
