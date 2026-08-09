import pandas as pd

from twinrag.detection import SensorModel, SensorSpec, StatisticalDetector


TIMESTAMPS = [hour * 3600 for hour in range(25)]

ASSETS = [str(100 + index) for index in range(10)]

QUIET_SENSORS = SensorModel(
    specs={"pressure": SensorSpec(absolute=0.01)},
    seed=0,
    noise_enabled=False,
)


def _frame(offsets=None):
    offsets = offsets or {}

    rows = []

    for asset in ASSETS:
        for timestamp in TIMESTAMPS:
            rows.append(
                {
                    "timestamp_s": timestamp,
                    "asset_id": asset,
                    "asset_type": "node",
                    "parameter": "pressure",
                    # A mild diurnal wobble, so MAD is not degenerate.
                    "value": 50.0
                    + 0.5 * (timestamp / 3600 % 6)
                    + offsets.get(asset, {}).get(timestamp, 0.0),
                    "scenario": "test",
                }
            )

    return pd.DataFrame(rows)


def _detector(**kwargs):
    options = {
        "sensor_model": QUIET_SENSORS,
        "score_threshold": 4.0,
        "min_assets": 3,
    }
    options.update(kwargs)

    return StatisticalDetector(**options)


def test_needs_no_baseline():
    """The whole point: it runs with no reference dataset at all."""

    report = _detector().detect(_frame())

    assert report.detector == "statistical"


def test_a_short_sharp_excursion_is_flagged():
    offsets = {
        asset: {timestamp: 30.0 for timestamp in TIMESTAMPS[8:11]}
        for asset in ASSETS[:4]
    }

    report = _detector().detect(_frame(offsets))

    assert report.detected
    assert report.first_detection_s == 8 * 3600


def test_a_constant_asset_is_not_flagged_on_rounding():
    """
    Two Net3 nodes are effectively constant across a normal day. Dividing
    by their near-zero spread would flag them on nothing at all, so the
    sensor noise floor has to win.
    """

    frame = _frame()
    frame["value"] = 50.0

    report = _detector().detect(frame)

    assert report.events == []


def test_expected_value_is_the_assets_own_median():
    report = _detector(score_threshold=0.5).detect(_frame())

    medians = (
        _frame().groupby("asset_id")["value"].median().to_dict()
    )

    for event in report.events[:20]:
        assert event.expected == medians[event.asset_id]


def test_a_long_fault_inverts_the_detection():
    """
    A fault occupying most of the run drags the very median it is measured
    against, until the *healthy* hours become the outliers. The detector
    still fires -- it just points at the wrong window.

    This is the structural weakness of judging an asset against its own
    history, and the reason the twin residual is the primary method rather
    than this one. It is asserted here so the failure mode is a documented
    property rather than a surprise in the evaluation numbers.
    """

    spike_window = TIMESTAMPS[8:10]
    long_window = TIMESTAMPS[2:22]

    spike = {
        asset: {timestamp: 8.0 for timestamp in spike_window}
        for asset in ASSETS[:4]
    }

    sustained = {
        asset: {timestamp: 8.0 for timestamp in long_window}
        for asset in ASSETS[:4]
    }

    on_spike = _detector().detect(_frame(spike))

    assert on_spike.detected
    assert {event.timestamp_s for event in on_spike.events} <= set(spike_window)

    on_sustained = _detector().detect(_frame(sustained))

    flagged = {event.timestamp_s for event in on_sustained.events}

    assert on_sustained.detected, "it fires, but on the healthy hours"
    assert not flagged & set(long_window), "the real fault window is missed"
    assert flagged <= set(TIMESTAMPS) - set(long_window)
