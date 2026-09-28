"""
Sensor model.

The simulator is deterministic: replaying the same network twice yields
bit-identical numbers, so the residual between a fault run and the
baseline is *exactly* zero before the fault fires. A detector evaluated
against that would score perfectly at any threshold above zero while
learning nothing — the absence of false positives would be an artifact
of the simulator, not a property of the detector.

Real instruments are noisy and sparse. This module puts both back:

    noise    -- what the twin predicts is not what a transducer reads
    coverage -- a real network is instrumented at a handful of points,
                not at all 97 nodes

Noise is drawn from a seeded generator over a canonically sorted frame,
so a given ``seed`` reproduces the same readings run after run.
"""

from dataclasses import dataclass

import numpy as np


#: Absolute noise floor plus a proportional term, per parameter.
#:
#: Distribution-network pressure transducers are typically quoted around
#: 0.25-0.5% of full scale; against Net3's ~93 m ceiling that is roughly
#: 0.3 m, which is the default here. Electromagnetic flow meters are
#: usually specified as a percentage of reading, hence the relative term.
DEFAULT_NOISE = {
    "pressure": (0.30, 0.0),
    "flowrate": (0.001, 0.01),
    "demand": (0.0005, 0.01),
}


@dataclass
class SensorSpec:
    """
    Noise scale for one measured parameter.

    The standard deviation of a reading is
    ``absolute + relative * |true value|``.
    """

    absolute: float = 0.0
    relative: float = 0.0

    def scale(self, values):
        """
        Per-reading noise standard deviation.
        """

        return self.absolute + self.relative * np.abs(values)


class SensorModel:
    """
    Turns simulated truth into what instruments would have reported.
    """

    def __init__(
        self,
        specs: dict | None = None,
        sensor_assets=None,
        seed: int = 0,
        noise_enabled: bool = True,
    ):
        """
        Parameters
        ----------
        specs
            ``{parameter: SensorSpec}``. Defaults to ``DEFAULT_NOISE``.
        sensor_assets
            Asset IDs that carry instruments. ``None`` means fully
            observed — every asset is measured.
        seed
            Seeds the noise draw. Same seed, same readings.
        noise_enabled
            Set ``False`` to pass truth through untouched. Useful for the
            noise-free ablation, which shows how much of a detector's
            apparent skill came from the simulator being deterministic.
        """

        if specs is None:
            specs = {
                parameter: SensorSpec(absolute, relative)
                for parameter, (absolute, relative) in DEFAULT_NOISE.items()
            }

        self.specs = specs
        self.sensor_assets = set(sensor_assets) if sensor_assets else None
        self.seed = seed
        self.noise_enabled = noise_enabled

    def observe(self, dataset):
        """
        Apply coverage and noise to a long-format dataset.

        Returns a copy carrying two extra columns:

        ``truth``
            The simulated value before noise, kept so evaluation can
            separate detector error from measurement error.
        ``noise_std``
            The noise scale for that reading. Detectors normalise
            residuals by this, which is what makes a score threshold
            mean "standard deviations" rather than "metres".
        """

        frame = dataset.copy()

        frame["asset_id"] = frame["asset_id"].astype(str)
        # Results straight from the simulator are float32; pandas 3 refuses
        # to write float64 noise into them. CSV round-trips hid this.
        frame["value"] = frame["value"].astype("float64")

        if self.sensor_assets is not None:
            frame = frame[frame["asset_id"].isin(self.sensor_assets)].copy()

        frame["truth"] = frame["value"]
        frame["noise_std"] = 0.0

        # Canonical ordering, so the noise a reading receives depends on
        # the seed alone -- not on how the rows happened to be stacked.
        frame = frame.sort_values(
            ["parameter", "asset_id", "timestamp_s"],
            kind="stable",
        ).reset_index(drop=True)

        generator = np.random.default_rng(self.seed)

        for parameter, spec in sorted(self.specs.items()):
            mask = frame["parameter"] == parameter

            if not mask.any():
                continue

            truth = frame.loc[mask, "truth"].to_numpy(dtype=float)

            scale = np.asarray(spec.scale(truth), dtype=float)
            scale = np.broadcast_to(scale, truth.shape).copy()

            # A zero scale would make the normalised score infinite, so
            # keep a small floor even for a notionally perfect sensor.
            scale = np.maximum(scale, 1e-9)

            frame.loc[mask, "noise_std"] = scale

            if self.noise_enabled:
                frame.loc[mask, "value"] = truth + generator.normal(
                    loc=0.0,
                    scale=scale,
                )

        return frame

    def describe(self) -> dict:
        """
        Configuration, for recording alongside detection results.
        """

        return {
            "noise_enabled": self.noise_enabled,
            "seed": self.seed,
            "sensor_count": (
                len(self.sensor_assets)
                if self.sensor_assets is not None
                else None
            ),
            "specs": {
                parameter: {
                    "absolute": spec.absolute,
                    "relative": spec.relative,
                }
                for parameter, spec in sorted(self.specs.items())
            },
        }
