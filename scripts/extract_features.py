#!/usr/bin/env python3
"""Extract DINOv2 features for all Timothy plant images.

Processes RGB1 (side: 3 angles) + RGB2 (top: 1 view) through DINOv2-base
and saves 768-dim CLS token embeddings to HDF5 per experiment.

Usage:
    python scripts/extract_features.py --experiment exp01
    python scripts/extract_features.py --experiment all --batch_size 128

Designed to run on LUMI (CSC) with AMD MI250X GPUs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = Path("data")
FEATURE_DIR = Path("features")

EXP_DIRS = {
    "exp01": DATA_DIR / "2023-Timothy-01-Nonvernalized",
    "exp02": DATA_DIR / "2024-Timothy-02-Vernalized",
    "exp03": DATA_DIR / "2024-Timothy-03-Regrowth",
}

EXP_IMG_PREFIX = {
    "exp01": "112",
    "exp02": "115",
    "exp03": "115",
}


def load_and_convert_image(path: str) -> Image.Image:
    """Load image, convert RGBA to RGB on black background."""
    img = Image.open(path)
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])
        return background
    elif img.mode == "RGB":
        return img
    return img.convert("RGB")


def build_image_manifest(
    plant_meta: pd.DataFrame,
    tp_meta: pd.DataFrame,
    experiment: str,
) -> list[dict[str, Any]]:
    """Build manifest of all images for an experiment by scanning the filesystem."""
    exp_dir = EXP_DIRS[experiment]
    exp_plants = plant_meta[plant_meta["experiment"] == experiment]
    exp_rounds = tp_meta[tp_meta["experiment"] == experiment]["round_order"].astype(int).tolist()
    prefix = EXP_IMG_PREFIX[experiment]

    manifest = []

    for _, row in exp_plants.iterrows():
        tray_id = str(row["tray_id"])
        genotype = str(row["genotype"])
        treatment = str(row["treatment"])
        rep = str(row["rep"])
        vern_label = "Vernalized" if row["vernalized"] else "NonVernalized"
        plant_folder_name = f"{tray_id} - {rep}"

        # RGB1 (side view): 3 angles
        for angle in ["000", "120", "240"]:
            rgb1_dir = (
                exp_dir / "RGB1_Img" / "FEC" / genotype / treatment
                / vern_label / plant_folder_name / angle
            )
            if not rgb1_dir.exists():
                continue
            for round_num in exp_rounds:
                fname = f"{prefix}-{round_num}-{tray_id}-RGB1-{angle}-FishEyeCorrected.png"
                fpath = rgb1_dir / fname
                if fpath.exists():
                    manifest.append({
                        "tray_id": tray_id,
                        "round": round_num,
                        "view_key": f"side_{angle}",
                        "path": str(fpath),
                    })

        # RGB2 (top view): no angle subdirectory
        rgb2_dir = (
            exp_dir / "RGB2_Img" / "FEC" / genotype / treatment
            / vern_label / plant_folder_name
        )
        if not rgb2_dir.exists():
            continue
        for round_num in exp_rounds:
            fname = f"{prefix}-{round_num}-{tray_id}-RGB2-FishEyeCorrected.png"
            fpath = rgb2_dir / fname
            if fpath.exists():
                manifest.append({
                    "tray_id": tray_id,
                    "round": round_num,
                    "view_key": "top",
                    "path": str(fpath),
                })

    return manifest


class PlantImageDataset(Dataset):
    def __init__(self, manifest: list[dict[str, Any]], transform) -> None:
        self.manifest = manifest
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.manifest[idx]
        img = load_and_convert_image(record["path"])
        img_tensor = self.transform(img)
        return img_tensor, record


class DINOv2Backbone(nn.Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        self.model.set_grad_enabled(False) if hasattr(self.model, 'set_grad_enabled') else None
        self.device = device

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    @torch.no_grad()
    def extract(self, batch: torch.Tensor) -> np.ndarray:
        batch = batch.to(self.device)
        outputs = self.model(pixel_values=batch)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def process_experiment(
    experiment: str,
    plant_meta: pd.DataFrame,
    tp_meta: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: str,
) -> None:
    output_path = FEATURE_DIR / f"dinov2_features_{experiment}.h5"

    processed = set()
    if output_path.exists():
        with h5py.File(output_path, "r") as f:
            processed = set(f.keys())

    manifest = build_image_manifest(plant_meta, tp_meta, experiment)
    if processed:
        before = len(manifest)
        manifest = [m for m in manifest if m["tray_id"] not in processed]
        print(f"  Resuming: {len(processed)} plants done, {before - len(manifest)} images skipped")

    if not manifest:
        print(f"  {experiment}: all images already processed!")
        return

    print(f"  {experiment}: processing {len(manifest)} images...")

    backbone = DINOv2Backbone(device)
    dataset = PlantImageDataset(manifest, transform=backbone.preprocess)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(output_path, "a")
    total = 0
    start = time.time()

    try:
        for batch_imgs, batch_meta in tqdm(dataloader, desc=f"[{experiment}]"):
            cls_features = backbone.extract(batch_imgs)

            for i in range(cls_features.shape[0]):
                tray_id = batch_meta["tray_id"][i]
                round_num = str(int(batch_meta["round"][i]))
                view_key = batch_meta["view_key"][i]

                if tray_id not in h5_file:
                    h5_file.create_group(tray_id)
                if round_num not in h5_file[tray_id]:
                    h5_file[tray_id].create_group(round_num)

                group = h5_file[tray_id][round_num]
                if view_key not in group:
                    group.create_dataset(view_key, data=cls_features[i], dtype="float32")
                    total += 1

                if total % 500 == 0:
                    h5_file.flush()
    finally:
        h5_file.close()

    elapsed = time.time() - start
    print(f"  {experiment}: {total} features in {elapsed:.1f}s -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DINOv2 features for Timothy")
    parser.add_argument("--experiment", default="all", choices=["exp01", "exp02", "exp03", "all"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    plant_meta = pd.read_csv(DATA_DIR / "plant_metadata.csv")
    tp_meta = pd.read_csv(DATA_DIR / "timepoint_metadata.csv")

    experiments = list(EXP_DIRS.keys()) if args.experiment == "all" else [args.experiment]

    for exp in experiments:
        print(f"\n=== {exp} ===")
        process_experiment(exp, plant_meta, tp_meta, args.batch_size, args.num_workers, device)

    print("\nDone!")


if __name__ == "__main__":
    main()
