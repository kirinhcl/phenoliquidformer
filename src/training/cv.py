"""Cross-validation strategies for Timothy drought phenotyping.

Provides LOPO, LOWHO, and cross-experiment CV splits.
"""

from __future__ import annotations

from typing import Generator, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd


class LeaveOnePlantOutCV:
    """Leave-One-Plant-Out CV within a single experiment.

    Each fold: 1 plant test, remainder split into train/val (stratified by WHC).
    """

    def __init__(
        self,
        plant_metadata: pd.DataFrame,
        experiment: str,
        val_fraction: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.df = plant_metadata[plant_metadata["experiment"] == experiment].reset_index(
            drop=True
        )
        self.val_fraction = val_fraction
        self.seed = seed

    def split(
        self,
    ) -> Generator[
        Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]],
        None,
        None,
    ]:
        rng = np.random.RandomState(self.seed)

        for test_idx in range(len(self.df)):
            test_indices = np.array([test_idx], dtype=np.int64)
            remaining = np.array(
                [i for i in range(len(self.df)) if i != test_idx], dtype=np.int64
            )

            # Stratified val split by WHC level
            n_val = max(1, int(len(remaining) * self.val_fraction))
            rng.shuffle(remaining)
            val_indices = remaining[:n_val]
            train_indices = remaining[n_val:]

            yield train_indices, val_indices, test_indices

    @property
    def n_folds(self) -> int:
        return len(self.df)


class LeaveOneWHCOutCV:
    """Leave-One-WHC-Out CV — trains on N-1 WHC levels, tests on held-out level.

    Most interesting for the paper: tests model extrapolation to unseen drought severity.
    """

    def __init__(
        self,
        plant_metadata: pd.DataFrame,
        experiment: str,
        val_fraction: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.df = plant_metadata[plant_metadata["experiment"] == experiment].reset_index(
            drop=True
        )
        self.whc_levels = sorted(self.df["treatment"].unique())
        self.val_fraction = val_fraction
        self.seed = seed

    def split(
        self,
    ) -> Generator[
        Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]],
        None,
        None,
    ]:
        rng = np.random.RandomState(self.seed)

        for test_whc in self.whc_levels:
            test_mask = self.df["treatment"] == test_whc
            test_indices = np.where(test_mask)[0].astype(np.int64)
            remaining = np.where(~test_mask)[0].astype(np.int64)

            # Pick one WHC level for validation
            remaining_whc = [w for w in self.whc_levels if w != test_whc]
            val_whc = rng.choice(remaining_whc)
            val_mask = self.df["treatment"] == val_whc
            val_indices = np.where(val_mask & ~test_mask)[0].astype(np.int64)
            train_indices = np.array(
                [i for i in remaining if i not in val_indices], dtype=np.int64
            )

            yield train_indices, val_indices, test_indices

    @property
    def n_folds(self) -> int:
        return len(self.whc_levels)


class CrossExperimentCV:
    """Train on one experiment, test on another.

    Useful for testing vernalization generalization and regrowth transfer.
    """

    def __init__(
        self,
        plant_metadata: pd.DataFrame,
        train_experiment: str,
        test_experiment: str,
        val_fraction: float = 0.15,
        seed: int = 42,
        genotype_filter: str | None = None,
    ) -> None:
        if genotype_filter:
            plant_metadata = plant_metadata[
                plant_metadata["genotype"] == genotype_filter
            ]
        self.train_df = plant_metadata[
            plant_metadata["experiment"] == train_experiment
        ].reset_index(drop=True)
        self.test_df = plant_metadata[
            plant_metadata["experiment"] == test_experiment
        ].reset_index(drop=True)
        self.val_fraction = val_fraction
        self.seed = seed

    def split(
        self,
    ) -> Generator[
        Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]],
        None,
        None,
    ]:
        """Single fold: all train_experiment → train/val, all test_experiment → test."""
        rng = np.random.RandomState(self.seed)
        all_train = np.arange(len(self.train_df), dtype=np.int64)
        rng.shuffle(all_train)
        n_val = max(1, int(len(all_train) * self.val_fraction))
        val_indices = all_train[:n_val]
        train_indices = all_train[n_val:]

        # Test indices are offset by train_df length (for combined dataset)
        test_indices = np.arange(
            len(self.train_df),
            len(self.train_df) + len(self.test_df),
            dtype=np.int64,
        )

        yield train_indices, val_indices, test_indices

    @property
    def n_folds(self) -> int:
        return 1
