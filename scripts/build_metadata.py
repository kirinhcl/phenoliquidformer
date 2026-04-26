"""Build unified metadata CSVs from Timothy experiment Excel files.

Outputs:
    data/plant_metadata.csv — One row per plant with genotype, treatment, etc.
    data/timepoint_metadata.csv — Round-to-DAS mapping per experiment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EXPERIMENTS = {
    "exp01": {
        "dir": "2023-Timothy-01-Nonvernalized",
        "label": "Nonvernalized",
        "vernalized": False,
    },
    "exp02": {
        "dir": "2024-Timothy-02-Vernalized",
        "label": "Vernalized",
        "vernalized": True,
    },
    "exp03": {
        "dir": "2024-Timothy-03-Regrowth",
        "label": "Regrowth",
        "vernalized": True,
    },
}


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and string values."""
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    return df


def _plant_id_to_tray_id(plant_id: str) -> str:
    """Convert Plant ID to Tray ID format used in image filenames.

    23_UpSa01_001 → 23_UpSa01_Timothy_001
    23_F02_001    → 24_F2_Timothy_001
    """
    # Exp01 pattern: 23_UpSa01_NNN
    m = re.match(r"(23_UpSa01)_(\d+)$", plant_id)
    if m:
        return f"{m.group(1)}_Timothy_{m.group(2)}"
    # Exp02/03 pattern: 23_F02_NNN → 24_F2_Timothy_NNN
    m = re.match(r"23_F02_(\d+)$", plant_id)
    if m:
        return f"24_F2_Timothy_{m.group(1)}"
    return plant_id


def build_timepoint_metadata() -> pd.DataFrame:
    """Build round → DAS mapping for each experiment from DigBio files."""
    records = []
    for exp_id, exp_info in EXPERIMENTS.items():
        exp_dir = DATA_DIR / exp_info["dir"]
        digbio_file = list(exp_dir.glob("DigBio_Timothy-*.xlsx"))[0]
        df = _strip_cols(pd.read_excel(digbio_file))

        # Extract unique (round_order, das, date) tuples
        round_info = (
            df[["Round Order", "DAS", "Measuring Date"]]
            .dropna(subset=["Round Order", "DAS"])
            .drop_duplicates(subset=["Round Order"])
            .sort_values("Round Order")
        )
        for _, row in round_info.iterrows():
            records.append(
                {
                    "experiment": exp_id,
                    "round_order": int(row["Round Order"]),
                    "das": int(row["DAS"]),
                    "measuring_date": str(row["Measuring Date"])[:10],
                }
            )

    return pd.DataFrame(records)


def build_plant_metadata() -> pd.DataFrame:
    """Build plant-level metadata from DigBio + EndPoint files."""
    all_plants = []

    for exp_id, exp_info in EXPERIMENTS.items():
        exp_dir = DATA_DIR / exp_info["dir"]

        # --- DigBio: get plant list and time-series info ---
        digbio_file = list(exp_dir.glob("DigBio_Timothy-*.xlsx"))[0]
        df = _strip_cols(pd.read_excel(digbio_file))

        # Plant-level info from first occurrence
        plant_rows = df.drop_duplicates(subset=["Plant ID"]).set_index("Plant ID")

        # Count timepoints per plant
        tp_counts = df.groupby("Plant ID")["Round Order"].nunique()

        # --- EndPoint: harvest data ---
        endpoint_file = list(exp_dir.glob("EndPoint_Timothy-*_Weight+Flowering.xlsx"))[0]
        ep = _strip_cols(pd.read_excel(endpoint_file))

        # EndPoint uses Plant ID for exp01 but Tray ID format for exp02/03
        # Build a lookup keyed by both plant_id and tray_id
        fw_col = next((c for c in ep.columns if "Fresh Weight" in c), None)
        dw_col = next((c for c in ep.columns if "Dry Weight" in c), None)
        flower_col = next((c for c in ep.columns if "Flower" in c), None)

        ep_lookup: dict[str, pd.Series] = {}
        for _, ep_row in ep.iterrows():
            ep_pid = str(ep_row["Plant ID"]).strip()
            ep_lookup[ep_pid] = ep_row

        for plant_id in plant_rows.index:
            plant_id = str(plant_id).strip()
            row = plant_rows.loc[plant_id]

            rec = {
                "plant_id": plant_id,
                "tray_id": _plant_id_to_tray_id(plant_id),
                "experiment": exp_id,
                "experiment_label": exp_info["label"],
                "vernalized": exp_info["vernalized"],
                "genotype": str(row.get("Genotype", "")).strip(),
                "treatment": str(row.get("Treatment", "")).strip(),
                "whc_level": None,
                "rep": str(row.get("Replicate", row.get("Rep", ""))).strip(),
                "num_timepoints": int(tp_counts.get(plant_id, 0)),
                "fw_g": None,
                "dw_g": None,
                "flowering": None,
            }

            # Parse WHC level as float
            whc_match = re.search(r"WHC-(\d+)", rec["treatment"])
            if whc_match:
                rec["whc_level"] = int(whc_match.group(1)) / 100.0

            # Endpoint data — match by plant_id or tray_id
            tray_id = rec["tray_id"]
            ep_row = ep_lookup.get(plant_id)
            if ep_row is None:
                ep_row = ep_lookup.get(tray_id)
            if ep_row is not None:
                if fw_col:
                    rec["fw_g"] = ep_row.get(fw_col)
                if dw_col:
                    rec["dw_g"] = ep_row.get(dw_col)
                if flower_col:
                    rec["flowering"] = ep_row.get(flower_col)

            all_plants.append(rec)

    result = pd.DataFrame(all_plants)
    # Convert types
    result["whc_level"] = pd.to_numeric(result["whc_level"], errors="coerce")
    result["fw_g"] = pd.to_numeric(result["fw_g"], errors="coerce")
    result["dw_g"] = pd.to_numeric(result["dw_g"], errors="coerce")
    return result


def main() -> None:
    print("Building timepoint metadata...")
    tp_meta = build_timepoint_metadata()
    tp_path = DATA_DIR / "timepoint_metadata.csv"
    tp_meta.to_csv(tp_path, index=False)
    print(f"  Saved {len(tp_meta)} rows → {tp_path}")
    for exp_id in EXPERIMENTS:
        exp_tp = tp_meta[tp_meta["experiment"] == exp_id]
        print(f"  {exp_id}: {len(exp_tp)} timepoints, DAS {exp_tp['das'].min()}-{exp_tp['das'].max()}")

    print("\nBuilding plant metadata...")
    plant_meta = build_plant_metadata()
    plant_path = DATA_DIR / "plant_metadata.csv"
    plant_meta.to_csv(plant_path, index=False)
    print(f"  Saved {len(plant_meta)} rows → {plant_path}")
    for exp_id in EXPERIMENTS:
        exp_plants = plant_meta[plant_meta["experiment"] == exp_id]
        genotypes = exp_plants["genotype"].value_counts().to_dict()
        treatments = sorted(exp_plants["treatment"].unique())
        has_endpoint = exp_plants["fw_g"].notna().sum()
        print(f"  {exp_id}: {len(exp_plants)} plants, genotypes={genotypes}, "
              f"treatments={treatments}, endpoint_data={has_endpoint}")


if __name__ == "__main__":
    main()
