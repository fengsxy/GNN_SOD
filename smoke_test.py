"""Offline CPU integration test with synthetic images, not an accuracy benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parent


def run(work_dir, size=64):
    work = Path(work_dir).resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError("Smoke-test work directory must be empty")
    work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for i in range(3):
        image = Image.new("RGB", (80 + i * 3, 56 + i * 5), (30, 70, 90))
        ImageDraw.Draw(image).rectangle((15, 12, 45, 35), fill=(200, 170, 60))
        mask = Image.new("L", image.size)
        if i != 1:  # Also exercise an empty target.
            ImageDraw.Draw(mask).rectangle((15, 12, 45, 35), fill=255)
        image.save(work / f"image_{i}.png")
        mask.save(work / f"mask_{i}.png")
    (work / "train.txt").write_text("image_0.png mask_0.png\nimage_1.png mask_1.png\n")
    (work / "val.txt").write_text("image_2.png mask_2.png\n")
    (work / "predict.txt").write_text("image_2.png\n")

    def command(script, *args):
        subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True, cwd=ROOT)

    for split in ("train", "val", "predict"):
        command("pre_process.py", "--manifest", work / f"{split}.txt", "--out-dir", work / split,
                "--weights", "none", "--size", size, "--segments", 50, 100, "--device", "cpu")
    command("Main.py", "--data", work / "train", "--val-data", work / "val",
            "--out-dir", work / "model", "--epochs", 1, "--batch-size", 2)
    command("Main.py", "--data", work / "train", "--val-data", work / "val",
            "--out-dir", work / "model", "--epochs", 2, "--batch-size", 1,
            "--resume", work / "model/latest.pt")
    command("predict.py", "--data", work / "predict", "--checkpoint", work / "model/latest.pt",
            "--out-dir", work / "predictions")
    state = torch.load(work / "model/latest.pt", map_location="cpu", weights_only=True)
    assert state["epoch"] == 2
    assert all(np.isfinite(row["train_loss"]) and np.isfinite(row["val_mae"]) for row in state["history"])
    assert (work / "model/best.pt").is_file()
    with Image.open(work / "predictions/000000.png") as prediction:
        assert prediction.mode == "L" and prediction.size == (86, 66)
    summary = {"status": "passed", "device": "cpu", "size": size, "segments": [50, 100],
               "knn": 32, "encoder_weights": "random, seed=0 (smoke only)",
               "train_images": 2, "validation_images": 1, "prediction_images": 1,
               "train_batch_sizes": [2, 1], "epochs_including_resume": 2,
               "history": state["history"], "seconds": round(time.perf_counter() - started, 2)}
    (work / "smoke_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", default="runs/smoke")
    p.add_argument("--size", type=int, default=64)
    args = p.parse_args()
    run(args.work_dir, args.size)
