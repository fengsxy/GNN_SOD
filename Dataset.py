"""Load tensor-only caches; each sample is a list of graph scales."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from runtime import check_config


class MyDataset(Dataset):
    def __init__(self, path, require_target=True):
        self.path = Path(path)
        manifest = self.path / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing {manifest}; build a new cache with pre_process.py")
        document = json.loads(manifest.read_text(encoding="utf-8"))
        self.config = document["config"]
        self.samples = document["samples"]
        self.require_target = require_target
        if self.config.get("schema_version") != 1 or not self.samples:
            raise ValueError("Unsupported or empty cache")
        if require_target and any(not s["mask_path"] for s in self.samples):
            raise ValueError("Training/validation requires a mask for every image")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        entry = self.samples[index]
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("Invalid cache filename")
        sample = torch.load(self.path / entry["file"], map_location="cpu", weights_only=True)
        check_config(self.config, sample["config"])
        if len(sample["graphs"]) != len(self.config["segments"]) or len(sample["cnn"]) != 4:
            raise ValueError("Cache has the wrong number of graph scales or CNN features")
        graphs = [Data(**graph) for graph in sample["graphs"]]
        for i, feature in enumerate(sample["cnn"]):
            setattr(graphs[0], f"cnn_{i}", feature.unsqueeze(0))
        if self.require_target:
            if sample["target"] is None:
                raise ValueError("Training/validation sample has no target")
            graphs[0].y = sample["target"].unsqueeze(0)
        graphs[0].original_size = torch.tensor([sample["original_size"]])
        graphs[0].sample_id = sample["sample_id"]
        return graphs
