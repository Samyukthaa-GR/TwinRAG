"""
The building's instrumentation as a detection ``SensorModel``.

The layout lists which assets carry an instrument; this module adds how
good those instruments are, so detection on the building sees exactly
what a real building management system would: noisy readings, and only
at the metered points.
"""

from twinrag.detection.sensors import SensorModel, SensorSpec


#: Instrument noise for the building, as ``(absolute, relative)``
#: standard deviations in SI units (m, m3/s):
#:
#: pressure -- a 0-10 bar transducer at 0.25% of full scale, ~0.25 m.
#: flowrate -- a domestic flow meter: 0.002 L/s floor plus 2% of reading.
#:
#: Demand is not metered at taps in a real building (and a simulated
#: leak *is* extra demand), so it is deliberately absent.
BUILDING_SENSOR_NOISE = {
    "pressure": (0.25, 0.0),
    "flowrate": (2e-6, 0.02),
}


def building_sensor_model(layout, seed: int = 0, noise_enabled: bool = True):
    """
    ``SensorModel`` restricted to the building's instrumented assets.
    """

    specs = {
        parameter: SensorSpec(absolute, relative)
        for parameter, (absolute, relative) in BUILDING_SENSOR_NOISE.items()
    }

    return SensorModel(
        specs=specs,
        sensor_assets={sensor["asset_id"] for sensor in layout["sensors"]},
        seed=seed,
        noise_enabled=noise_enabled,
    )
