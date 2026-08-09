"""
Twin-residual detection.

Compares what the instruments report against what the digital twin says
the network should be doing at that moment, and flags readings the two
cannot agree on:

    score = |observed - twin_expected| / sensor_noise_std

Normalising by the sensor's own noise scale is what makes the threshold
portable: it is expressed in standard deviations, so one setting works
across pressure in metres and flow in cubic metres per second without
per-parameter tuning.

This is the method the whole project is built to support. A twin exists
precisely so there is an expectation to compare against, and unlike the
purely statistical fallback it can catch a fault that holds the network
at a wrong-but-steady state, where an asset's own history looks calm.

Its cost is a dependency: the expectation is only as good as the twin.
``StatisticalDetector`` covers the case where no trustworthy reference
run is available.
"""

import numpy as np

from .base import AnomalyDetector
from .sensors import SensorModel


class ResidualDetector(AnomalyDetector):
    """
    Flags readings that disagree with the twin's prediction.
    """

    name = "residual"

    def __init__(
        self,
        baseline,
        sensor_model: SensorModel | None = None,
        parameters=None,
        score_threshold: float = 4.0,
        min_assets: int = 3,
        min_residual: dict | None = None,
    ):
        """
        Parameters
        ----------
        baseline
            Long-format dataset from the unfaulted network — the twin's
            expectation. Its ``truth`` column is used when present, so a
            baseline that has been through a sensor model still
            contributes its noise-free values.
        sensor_model
            Applied to the observed dataset before comparison. Defaults
            to a standard-noise model.
        min_residual
            Optional ``{parameter: value}`` floor on the raw deviation.
            A very precise sensor can make a hydraulically irrelevant
            wobble look statistically enormous; this suppresses that
            without weakening the score threshold.
        """

        super().__init__(
            parameters=parameters,
            score_threshold=score_threshold,
            min_assets=min_assets,
        )

        self.sensor_model = sensor_model or SensorModel()
        self.min_residual = dict(min_residual or {})

        self.baseline = self._prepare_baseline(baseline)

    def _prepare_baseline(self, baseline):
        """
        Reduce the reference run to an expectation lookup.
        """

        column = "truth" if "truth" in baseline.columns else "value"

        frame = baseline[
            ["timestamp_s", "asset_id", "parameter", column]
        ].copy()

        frame = frame.rename(columns={column: "expected"})

        # Asset IDs are numeric strings in Net3 ("101", "10"), which
        # read_csv happily turns into integers. Compare as text on both
        # sides or the join silently matches nothing.
        frame["asset_id"] = frame["asset_id"].astype(str)

        return frame

    def detect(self, dataset):
        """
        Detect anomalies in one scenario.
        """

        observed = self.sensor_model.observe(dataset)

        observed = observed[observed["parameter"].isin(self.parameters)]

        if observed.empty:
            raise ValueError(
                f"No readings for parameters {list(self.parameters)}; "
                f"dataset has {sorted(dataset['parameter'].unique())}."
            )

        scenario = self._scenario_name(dataset)

        merged = observed.merge(
            self.baseline,
            on=["timestamp_s", "asset_id", "parameter"],
            how="inner",
        )

        if merged.empty:
            raise ValueError(
                "Observed data and baseline share no "
                "(timestamp, asset, parameter) keys."
            )

        merged = merged.rename(columns={"value": "observed"})

        merged["residual"] = merged["observed"] - merged["expected"]

        merged["score"] = np.abs(merged["residual"]) / merged["noise_std"]

        if self.min_residual:
            floor = (
                merged["parameter"]
                .map(self.min_residual)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

            below_floor = np.abs(merged["residual"].to_numpy()) < floor

            merged.loc[below_floor, "score"] = 0.0

        events = self._events_from_flags(merged, scenario)

        timeline = sorted(observed["timestamp_s"].unique().tolist())

        return self._report(
            scenario=scenario,
            events=events,
            timeline=timeline,
            extra_config={"sensors": self.sensor_model.describe()},
        )
