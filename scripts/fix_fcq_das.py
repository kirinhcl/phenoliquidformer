#!/usr/bin/env python3
"""Reconcile FCQ_Timothy-XX.xlsx DAS column against canonical timepoint_metadata.

The FCQ spreadsheets occasionally contain rows whose DAS column disagrees with
the Measuring Date + experiment start date. `Round Order` and `Measuring Date`
are consistently correct; `DAS` is the only mislabeled column. This script
writes a `_fixed` sidecar with DAS recomputed from date, leaving every other
cell byte-identical.

Model training and saliency are unaffected because src/data/dataset.py maps
fluorescence by `Round Order` (see dataset.py:166-170), not DAS. This fix
targets downstream analyses (Spearman rho, paper stats) that pivot on DAS.

Usage:
    python scripts/fix_fcq_das.py --experiment exp01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FCQ_PATHS = {
    "exp01": Path("data/2023-Timothy-01-Nonvernalized/FCQ_Timothy-01.xlsx"),
    "exp02": Path("data/2024-Timothy-02-Vernalized/FCQ_Timothy-02.xlsx"),
    "exp03": Path("data/2024-Timothy-03-Regrowth/FCQ_Timothy-03.xlsx"),
}
TIMEPOINTS = Path("data/timepoint_metadata.csv")


def canonical_das_map(experiment: str) -> dict[str, int]:
    tp = pd.read_csv(TIMEPOINTS)
    tp = tp[tp["experiment"] == experiment].copy()
    tp["md"] = pd.to_datetime(tp["measuring_date"], format="mixed", errors="coerce")
    tp = tp.dropna(subset=["md"])
    tp["md"] = tp["md"].dt.strftime("%Y-%m-%d")
    grouped = tp.groupby("md")["das"].first()
    return grouped.to_dict()


def fix_fcq(experiment: str) -> None:
    src = FCQ_PATHS[experiment]
    if not src.exists():
        raise FileNotFoundError(f"FCQ not found: {src}")

    dst = src.with_name(src.stem + "_fixed" + src.suffix)

    df = pd.read_excel(src)
    original_cols = list(df.columns)

    date_col = next(c for c in df.columns if str(c).strip() == "Measuring Date")
    das_col = next(c for c in df.columns if str(c).strip() == "DAS")
    round_col = next(c for c in df.columns if str(c).strip() == "Round Order")

    date_str = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    canon = canonical_das_map(experiment)
    canon_das = date_str.map(canon)

    orig_das = df[das_col].astype("Int64")
    new_das = canon_das.astype("Int64")
    mismatch = orig_das.notna() & new_das.notna() & (orig_das != new_das)

    n_rows = int(mismatch.sum())
    if n_rows == 0:
        print(f"[{experiment}] no DAS mismatches; original file is already canonical.")
        return

    mismatch_summary = (
        df.loc[mismatch, [date_col, round_col, das_col]]
        .assign(canonical_das=new_das[mismatch].values)
        .drop_duplicates()
    )
    print(f"[{experiment}] correcting {n_rows} row(s):")
    print(mismatch_summary.to_string(index=False))

    df.loc[mismatch, das_col] = new_das[mismatch].astype("Int64").astype(int)
    df = df[original_cols]
    df.to_excel(dst, index=False)
    print(f"[{experiment}] wrote {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", choices=sorted(FCQ_PATHS), default="exp01",
        help="Which experiment's FCQ to reconcile (default: exp01)",
    )
    args = parser.parse_args()
    fix_fcq(args.experiment)


if __name__ == "__main__":
    main()
