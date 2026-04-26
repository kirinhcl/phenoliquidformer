#!/usr/bin/env python3
"""Run modality ablation experiments.

Trains 9 model variants (1 full + 4 single-modality + 4 drop-one-modality)
with LOWHO CV and compares test MAE.

Usage:
    python scripts/run_ablations.py --cv lowho
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ABLATIONS = {
    "full":        ["image", "fluor", "env", "vi"],
    "image_only":  ["image"],
    "fluor_only":  ["fluor"],
    "env_only":    ["env"],
    "vi_only":     ["vi"],
    "drop_image":  ["fluor", "env", "vi"],
    "drop_fluor":  ["image", "env", "vi"],
    "drop_env":    ["image", "fluor", "vi"],
    "drop_vi":     ["image", "fluor", "env"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/timothy.yaml")
    parser.add_argument("--cv", default="lowho", choices=["lopo", "lowho"])
    parser.add_argument("--output-base", default="results/ablation")
    args = parser.parse_args()

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    summary = {}

    for ablation_name, enabled_mods in ABLATIONS.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_name} | modalities: {enabled_mods}")
        print(f"{'='*60}", flush=True)

        ablation_dir = output_base / ablation_name
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # Create modified config
        cfg = dict(base_cfg)
        cfg["model"] = dict(cfg["model"])
        cfg["model"]["ablation"] = dict(cfg["model"].get("ablation", {}))
        cfg["model"]["ablation"]["enabled_modalities"] = enabled_mods

        tmp_cfg_path = ablation_dir / "config.yaml"
        with open(tmp_cfg_path, "w") as f:
            yaml.dump(cfg, f)

        # Run training
        cmd = [
            sys.executable, "scripts/train_timothy.py",
            "--config", str(tmp_cfg_path),
            "--cv", args.cv,
            "--output-dir", str(ablation_dir),
        ]
        print(f"Running: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {ablation_name}", flush=True)
            continue

        # Produced path: {ablation_dir}/exp01_{cv}/
        produced = ablation_dir / f"exp01_{args.cv}"
        summary_file = produced / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary[ablation_name] = json.load(f)
                summary[ablation_name]["modalities"] = enabled_mods

    # Save overall summary
    with open(output_base / f"ablation_summary_{args.cv}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print("\n\n" + "=" * 70)
    print(f"ABLATION SUMMARY ({args.cv})")
    print("=" * 70)
    print(f"{'Ablation':<15} {'Modalities':<35} {'MAE':<10} {'std':<10}")
    print("-" * 70)
    for name, data in summary.items():
        mods_str = ",".join(data.get("modalities", []))
        mae = data.get("mean_mae", 0)
        std = data.get("std_mae", 0)
        print(f"{name:<15} {mods_str:<35} {mae:<10.4f} {std:<10.4f}")


if __name__ == "__main__":
    main()
