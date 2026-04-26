"""PyTorch dataset for Timothy grass drought phenotyping multimodal time-series data.

Loads pre-extracted image features, fluorescence measurements, environment data,
vegetation indices, and digital biomass for plants across variable timepoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils.config import load_config

# Probe sheet names in Envdata files → canonical env feature order
ENV_PROBES = ["li1", "t1", "rh1", "t2", "rh2"]


class TimothyDroughtDataset(Dataset[Dict[str, Any]]):
    """Multimodal time-series dataset for Timothy grass drought phenotyping.

    Returns per-plant dict with:
        images: (T, V=4, D=768) float32
        image_mask: (T, V=4) bool
        fluorescence: (T, fluor_dim) float32
        fluor_mask: (T,) bool
        environment: (T, 5) float32
        vi: (T, vi_dim) float32
        temporal_positions: (T,) float32 — DAS values
        whc_target: float — WHC level (0.25-0.90)
        fw_target: float — Fresh weight (NaN if missing)
        dw_target: float — Dry weight (NaN if missing)
        digital_biomass: (T,) float32 — biomass trajectory
        biomass_mask: (T,) bool
        plant_id: str
        tray_id: str
        genotype: str
        treatment: str
        experiment: str
    """

    def __init__(self, config_path_or_cfg: Union[str, DictConfig]) -> None:
        if isinstance(config_path_or_cfg, str):
            self.cfg = load_config(config_path_or_cfg)
        else:
            self.cfg = config_path_or_cfg

        # Load metadata
        self.plant_meta = pd.read_csv(self.cfg.data.plant_metadata)
        self.tp_meta = pd.read_csv(self.cfg.data.timepoint_metadata)

        # Filter by experiment
        experiment = self.cfg.data.experiment
        if experiment != "all":
            self.plant_meta = self.plant_meta[
                self.plant_meta["experiment"] == experiment
            ].reset_index(drop=True)

        # Build round→DAS mapping per experiment
        self.round_to_das: dict[str, dict[int, int]] = {}
        for exp_id, grp in self.tp_meta.groupby("experiment"):
            self.round_to_das[str(exp_id)] = dict(
                zip(grp["round_order"].astype(int), grp["das"].astype(int))
            )

        # Determine max timepoints for current experiment(s)
        if experiment == "all":
            self.max_T = max(len(v) for v in self.round_to_das.values())
        else:
            self.max_T = len(self.round_to_das[experiment])

        # Temporal positions (DAS values) per experiment
        self.temporal_positions_map: dict[str, torch.Tensor] = {}
        for exp_id, rd_map in self.round_to_das.items():
            sorted_rounds = sorted(rd_map.keys())
            das_values = [rd_map[r] for r in sorted_rounds]
            self.temporal_positions_map[exp_id] = torch.tensor(
                das_values, dtype=torch.float32
            )

        # Round order lists per experiment (for indexing)
        self.round_lists: dict[str, list[int]] = {
            exp_id: sorted(rd_map.keys())
            for exp_id, rd_map in self.round_to_das.items()
        }

        # Load HDF5 features
        feature_dir = Path(self.cfg.data.feature_dir)
        self.h5_handles: dict[str, h5py.File] = {}
        for exp_id in self.round_to_das:
            h5_path = feature_dir / f"dinov2_features_{exp_id}.h5"
            if h5_path.exists():
                self.h5_handles[exp_id] = h5py.File(h5_path, "r")

        # Load tabular modality data
        self.fluor_normalize = getattr(self.cfg.data, "fluor_normalize", False)
        self.fluor_data, self.fluor_dim = self._load_fluorescence()
        self.env_data = self._load_environment()
        self.vi_data, self.vi_dim = self._load_vi()
        self.biomass_data = self._load_digital_biomass()

        # Compute biomass normalization stats (log1p + z-score)
        all_bm = [v for plant in self.biomass_data.values() for v in plant.values() if v > 0]
        if all_bm:
            import math
            log_bm = [math.log1p(v) for v in all_bm]
            self.biomass_mean = float(np.mean(log_bm))
            self.biomass_std = max(float(np.std(log_bm)), 1e-8)
        else:
            self.biomass_mean = 0.0
            self.biomass_std = 1.0

    def __len__(self) -> int:
        return len(self.plant_meta)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.plant_meta.iloc[idx]
        plant_id = str(row["plant_id"])
        tray_id = str(row["tray_id"])
        experiment = str(row["experiment"])
        rounds = self.round_lists[experiment]
        T = len(rounds)

        V, D = 4, self.cfg.model.encoder_output_dim

        images = torch.zeros((self.max_T, V, D), dtype=torch.float32)
        image_mask = torch.zeros((self.max_T, V), dtype=torch.bool)
        fluorescence = torch.zeros((self.max_T, self.fluor_dim), dtype=torch.float32)
        fluor_mask = torch.zeros(self.max_T, dtype=torch.bool)
        environment = torch.zeros((self.max_T, 5), dtype=torch.float32)
        vi = torch.zeros((self.max_T, self.vi_dim), dtype=torch.float32)
        digital_biomass = torch.zeros(self.max_T, dtype=torch.float32)
        biomass_mask = torch.zeros(self.max_T, dtype=torch.bool)

        # Temporal positions (padded to max_T)
        tp = self.temporal_positions_map[experiment]
        temporal_positions = torch.zeros(self.max_T, dtype=torch.float32)
        temporal_positions[:T] = tp

        # Load image features from HDF5
        view_keys = ["side_000", "side_120", "side_240", "top"]
        h5 = self.h5_handles.get(experiment)
        if h5 is not None and tray_id in h5:
            plant_group = h5[tray_id]
            for t_idx, round_num in enumerate(rounds):
                if str(round_num) in plant_group:
                    round_group = plant_group[str(round_num)]
                    for v_idx, vk in enumerate(view_keys):
                        if vk in round_group:
                            images[t_idx, v_idx] = torch.from_numpy(
                                round_group[vk][:]
                            )
                            image_mask[t_idx, v_idx] = True

        # Load fluorescence
        key = f"{experiment}:{plant_id}"
        if key in self.fluor_data:
            for round_num, vec in self.fluor_data[key].items():
                if round_num in rounds:
                    t_idx = rounds.index(round_num)
                    fluorescence[t_idx] = torch.from_numpy(vec)
                    fluor_mask[t_idx] = True

        # Load environment (global per experiment per round)
        env_key = experiment
        if env_key in self.env_data:
            for round_num, vec in self.env_data[env_key].items():
                if round_num in rounds:
                    t_idx = rounds.index(round_num)
                    environment[t_idx] = torch.from_numpy(vec)

        # Load vegetation indices
        if key in self.vi_data:
            for round_num, vec in self.vi_data[key].items():
                if round_num in rounds:
                    t_idx = rounds.index(round_num)
                    vi[t_idx] = torch.from_numpy(vec)

        # Load digital biomass (log1p + z-score normalized)
        if key in self.biomass_data:
            for round_num, val in self.biomass_data[key].items():
                if round_num in rounds:
                    t_idx = rounds.index(round_num)
                    log_val = np.log1p(max(val, 0.0))
                    digital_biomass[t_idx] = (log_val - self.biomass_mean) / self.biomass_std
                    biomass_mask[t_idx] = True

        # Clean NaNs
        torch.nan_to_num_(fluorescence, nan=0.0)
        torch.nan_to_num_(environment, nan=0.0)
        torch.nan_to_num_(vi, nan=0.0)
        torch.nan_to_num_(digital_biomass, nan=0.0)

        # Targets
        whc_target = float(row["whc_level"])
        fw_target = float(row["fw_g"]) if pd.notna(row["fw_g"]) else float("nan")
        dw_target = float(row["dw_g"]) if pd.notna(row["dw_g"]) else float("nan")

        # Active mask for this experiment's timepoints
        active_mask = torch.zeros(self.max_T, dtype=torch.bool)
        active_mask[:T] = True

        return {
            "images": images,
            "image_mask": image_mask,
            "fluorescence": fluorescence,
            "fluor_mask": fluor_mask,
            "environment": environment,
            "vi": vi,
            "temporal_positions": temporal_positions,
            "active_mask": active_mask,
            "whc_target": whc_target,
            "fw_target": fw_target,
            "dw_target": dw_target,
            "digital_biomass": digital_biomass,
            "biomass_mask": biomass_mask,
            "plant_id": plant_id,
            "tray_id": tray_id,
            "genotype": str(row["genotype"]),
            "treatment": str(row["treatment"]),
            "experiment": experiment,
        }

    # --- Data loading helpers ---

    def _get_experiment_files(self) -> dict[str, Path]:
        """Map experiment IDs to their data directories."""
        data_dir = Path(self.cfg.data.plant_metadata).parent
        return {
            "exp01": data_dir / "2023-Timothy-01-Nonvernalized",
            "exp02": data_dir / "2024-Timothy-02-Vernalized",
            "exp03": data_dir / "2024-Timothy-03-Regrowth",
        }

    def _load_fluorescence(
        self,
    ) -> tuple[dict[str, dict[int, npt.NDArray[np.float32]]], int]:
        """Load fluorescence data from FCQ Excel files for all experiments."""
        exp_dirs = self._get_experiment_files()
        fluor_data: dict[str, dict[int, npt.NDArray[np.float32]]] = {}
        fluor_dim = None

        # Determine which experiments to load
        experiments = (
            list(exp_dirs.keys())
            if self.cfg.data.experiment == "all"
            else [self.cfg.data.experiment]
        )

        fluor_means: npt.NDArray[np.float32] | None = None
        fluor_stds: npt.NDArray[np.float32] | None = None

        for exp_id in experiments:
            exp_dir = exp_dirs[exp_id]
            fcq_files = list(exp_dir.glob("FCQ_Timothy-*.xlsx"))
            if not fcq_files:
                continue
            df = pd.read_excel(fcq_files[0])
            df.columns = [str(c).strip() for c in df.columns]

            # Identify fluorescence columns (after metadata columns)
            exclude = {"Obs", "Camera Position", "Size"}
            meta_end = 20  # metadata columns end around index 20
            fluor_cols = [
                c
                for i, c in enumerate(df.columns)
                if i >= meta_end
                and c not in exclude
                and pd.api.types.is_numeric_dtype(df[c])
            ]

            # Enforce consistent fluor_dim from config (different experiments
            # may parse slightly different column counts from Excel)
            try:
                target_dim = int(self.cfg.model.modality.fluor_dim)
            except Exception:
                target_dim = 98
            if len(fluor_cols) > target_dim:
                fluor_cols = fluor_cols[:target_dim]
            elif len(fluor_cols) < target_dim:
                pass  # pad with zeros later in __getitem__

            if fluor_dim is None:
                fluor_dim = target_dim

            # Z-score normalization
            if self.fluor_normalize:
                mat = df[fluor_cols].values.astype(np.float32)
                fluor_means = np.nanmean(mat, axis=0)
                fluor_stds = np.nanstd(mat, axis=0)
                fluor_stds[fluor_stds < 1e-8] = 1.0

            df["Plant ID"] = df["Plant ID"].astype(str).str.strip()
            for _, row in df.iterrows():
                pid = str(row["Plant ID"])
                ro = row.get("Round Order")
                if pd.isna(ro):
                    continue
                ro = int(ro)
                vec = np.array(row[fluor_cols].tolist(), dtype=np.float32)
                np.nan_to_num(vec, copy=False, nan=0.0)
                if self.fluor_normalize and fluor_means is not None:
                    vec = (vec - fluor_means) / fluor_stds

                key = f"{exp_id}:{pid}"
                if key not in fluor_data:
                    fluor_data[key] = {}
                fluor_data[key][ro] = vec

        return fluor_data, fluor_dim or 99

    def _load_environment(self) -> dict[str, dict[int, npt.NDArray[np.float32]]]:
        """Load environment data aggregated per round per experiment.

        Timothy envdata has separate sheets per probe (rh1, t1, li1, t2, rh2).
        Returns {experiment: {round_order: np.array(5,)}}.
        """
        exp_dirs = self._get_experiment_files()
        env_data: dict[str, dict[int, npt.NDArray[np.float32]]] = {}

        experiments = (
            list(exp_dirs.keys())
            if self.cfg.data.experiment == "all"
            else [self.cfg.data.experiment]
        )

        for exp_id in experiments:
            exp_dir = exp_dirs[exp_id]
            env_files = list(exp_dir.glob("Envdata_Timothy-*_Probes.xlsx"))
            if not env_files:
                continue

            # Read all probe sheets and compute daily means
            probe_daily: dict[str, pd.Series] = {}
            for probe_name in ENV_PROBES:
                # Find matching sheet (may have _old suffix)
                sheet_candidates = [
                    f"{probe_name}_old",
                    f"{probe_name}",
                ]
                df = None
                for sheet in sheet_candidates:
                    try:
                        df = pd.read_excel(env_files[0], sheet_name=sheet)
                        break
                    except ValueError:
                        continue
                if df is None:
                    continue
                df.columns = [str(c).strip() for c in df.columns]
                df["Measuring Date"] = pd.to_datetime(df["Measuring Date"])
                df["date"] = df["Measuring Date"].dt.date
                probe_daily[probe_name] = df.groupby("date")["Value"].mean()

            # Map to rounds using timepoint metadata
            tp_exp = self.tp_meta[self.tp_meta["experiment"] == exp_id]
            exp_env: dict[int, npt.NDArray[np.float32]] = {}
            for _, tp_row in tp_exp.iterrows():
                round_order = int(tp_row["round_order"])
                date_str = str(tp_row["measuring_date"])[:10]
                try:
                    date = pd.to_datetime(date_str).date()
                except Exception:
                    continue
                vec = np.zeros(5, dtype=np.float32)
                for i, probe_name in enumerate(ENV_PROBES):
                    if probe_name in probe_daily and date in probe_daily[probe_name].index:
                        vec[i] = float(probe_daily[probe_name].loc[date])
                exp_env[round_order] = vec

            env_data[exp_id] = exp_env

        return env_data

    def _load_vi(
        self,
    ) -> tuple[dict[str, dict[int, npt.NDArray[np.float32]]], int]:
        """Load vegetation index data per plant per round."""
        exp_dirs = self._get_experiment_files()
        vi_data: dict[str, dict[int, npt.NDArray[np.float32]]] = {}
        vi_dim = None

        vi_cols = [
            "ExG", "GREENESS", "GLI", "GREEN_STRENGHT", "NGRVI", "VARI",
            "BG_RATIO", "CHROMA_BASE", "CHROMA_RATIO", "CHROMA_DIFFERENCE", "TGI",
        ]

        experiments = (
            list(exp_dirs.keys())
            if self.cfg.data.experiment == "all"
            else [self.cfg.data.experiment]
        )

        for exp_id in experiments:
            exp_dir = exp_dirs[exp_id]
            vi_files = list(exp_dir.glob("VegIndex_Timothy-*.xlsx"))
            if not vi_files:
                continue
            df = pd.read_excel(vi_files[0])
            df.columns = [str(c).strip() for c in df.columns]

            # Use available VI columns
            available_cols = [c for c in vi_cols if c in df.columns]
            if vi_dim is None:
                vi_dim = len(available_cols)

            pid_col = "Plant ID" if "Plant ID" in df.columns else "Tray ID"
            df[pid_col] = df[pid_col].astype(str).str.strip()

            for _, row in df.iterrows():
                pid = str(row[pid_col])
                ro = row.get("Round Order")
                if pd.isna(ro):
                    continue
                ro = int(ro)
                vec = np.array(row[available_cols].tolist(), dtype=np.float32)
                np.nan_to_num(vec, copy=False, nan=0.0)

                key = f"{exp_id}:{pid}"
                if key not in vi_data:
                    vi_data[key] = {}
                vi_data[key][ro] = vec

        return vi_data, vi_dim or 11

    def _load_digital_biomass(self) -> dict[str, dict[int, float]]:
        """Load digital biomass trajectory data per plant."""
        exp_dirs = self._get_experiment_files()
        biomass_data: dict[str, dict[int, float]] = {}

        experiments = (
            list(exp_dirs.keys())
            if self.cfg.data.experiment == "all"
            else [self.cfg.data.experiment]
        )

        for exp_id in experiments:
            exp_dir = exp_dirs[exp_id]
            db_files = list(exp_dir.glob("DigBio_Timothy-*.xlsx"))
            if not db_files:
                continue
            df = pd.read_excel(db_files[0])
            df.columns = [str(c).strip() for c in df.columns]

            biomass_col = "Digital Biomass"
            pid_col = "Plant ID"
            df[pid_col] = df[pid_col].astype(str).str.strip()

            for _, row in df.iterrows():
                pid = str(row[pid_col])
                ro = row.get("Round Order")
                bm = row.get(biomass_col)
                if pd.isna(ro) or pd.isna(bm):
                    continue

                key = f"{exp_id}:{pid}"
                if key not in biomass_data:
                    biomass_data[key] = {}
                biomass_data[key][int(ro)] = float(bm)

        return biomass_data

    def __del__(self) -> None:
        for h5 in self.h5_handles.values():
            h5.close()
