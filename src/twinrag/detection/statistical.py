"""
Unsupervised statistical detection.

The baseline-free fallback. Each asset is judged against *its own*
distribution over the run rather than against a twin prediction:

    score = |value - median| / max(1.4826 x MAD, sensor_noise_std)

The median and MAD are robust estimators, so the fault window itself
does not drag the reference it is being measured against. The noise floor
matters more than it looks: two of Net3's pressure nodes are essentially
constant across a normal day, and dividing by their near-zero spread
would flag them on nothing but rounding.

This detector exists to answer the obvious objection to residual
detection — "you only found it because you had the answer" — and to cover
real deployments where no trustworthy reference run exists. It is
expected to be the weaker of the two, for two structural reasons worth
stating rather than discovering later:

1. A six-hour fault occupies roughly a quarter of a 24-hour run, so the
   anomalous points are a large minority of the very sample defining
   "normal".
2. Demand in Net3 swings diurnally, so an asset's own spread is already
   wide; a modest fault hides inside it. At 30% severity a leak moves its
   target junction about 0.87 m, against a median per-asset daily spread
   of 0.88 m.

Where it wins is independence: it needs no reference run, so it still
works when the twin is mis-calibrated or the network has drifted away
from the model.
"""

import numpy as np

from .base import AnomalyDetector
from .sensors import SensorModel


#: Scales the median absolute deviation to a standard-deviation estimate
#: for normally distributed data.
MAD_TO_SIGMA = 1.4826


class StatisticalDetector(AnomalyDetector):
    """
    Flags readings that deviate from an asset's own typical level.
    """

    name = "statistical"

    def __init__(
        self,
        sensor_model: SensorModel | None = None,
        parameters=None,
        score_threshold: float = 4.0,
        min_assets: int = 3,
    ):
        super().__init__(
            parameters=parameters,
            score_threshold=score_threshold,
            min_assets=min_assets,
        )

        self.sensor_model = sensor_model or SensorModel()

    def detect(self, dataset):
        """
        Detect anomalies in one scenario, using no reference run.
        """

        observed = self.sensor_model.observe(dataset)

        observed = observed[
            observed["parameter"].isin(self.parameters)
        ].copy()

        if observed.empty:
            raise ValueError(
                f"No readings for parameters {list(self.parameters)}; "
                f"dataset has {sorted(dataset['parameter'].unique())}."
            )

        scenario = self._scenario_name(dataset)

        observed = observed.rename(columns={"value": "observed"})

        grouped = observed.groupby(
            ["parameter", "asset_id"],
            sort=False,
        )["observed"]

        observed["expected"] = grouped.transform("median")

        observed["residual"] = observed["observed"] - observed["expected"]

        absolute_deviation = observed["residual"].abs()

        # MAD per asset: the median of each reading's distance from its
        # asset's median. Computed via a second grouped transform on the
        # deviation column rather than an apply, to stay vectorised.
        observed["_abs_dev"] = absolute_deviation

        mad = observed.groupby(
            ["parameter", "asset_id"],
            sort=False,
        )["_abs_dev"].transform("median")

        scale = np.maximum(
            MAD_TO_SIGMA * mad.to_numpy(dtype=float),
            observed["noise_std"].to_numpy(dtype=float),
        )

        observed["score"] = absolute_deviation.to_numpy() / scale

        observed = observed.drop(columns=["_abs_dev"])

        events = self._events_from_flags(observed, scenario)

        timeline = sorted(observed["timestamp_s"].unique().tolist())

        return self._report(
            scenario=scenario,
            events=events,
            timeline=timeline,
            extra_config={"sensors": self.sensor_model.describe()},
        )
