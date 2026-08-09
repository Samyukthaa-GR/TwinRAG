import pandas as pd
import pytest

from twinrag.detection import ResidualDetector, SensorModel, SensorSpec


TIMESTAMPS = [hour * 3600 for hour in range(25)]

ASSETS = [str(100 + index) for index in range(10)]

QUIET_SENSORS = SensorModel(
    specs={"pressure": SensorSpec(absolute=1.0)},
    seed=0,
    noise_enabled=False,
)


def _frame(offsets=None, scenario="test"):
    """
    Ten pressure assets holding 50 m, with optional per-(asset, time)
    offsets applied on top.

    ``offsets`` maps ``asset_id -> {timestamp_s: delta}``.
    """

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
                    "value": 50.0 + offsets.get(asset, {}).get(timestamp, 0.0),
                    "scenario": scenario,
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

    return ResidualDetector(baseline=_frame(), **options)


def test_identical_run_produces_nothing():
    report = _detector().detect(_frame())

    assert report.events == []
    assert not report.detected


def test_deviation_above_threshold_is_flagged():
    window = [hour * 3600 for hour in range(8, 14)]

    offsets = {
        asset: {timestamp: 10.0 for timestamp in window}
        for asset in ASSETS[:4]
    }

    report = _detector().detect(_frame(offsets))

    assert report.detected
    assert report.first_detection_s == 8 * 3600

    incident = report.incidents[0]

    assert incident.detected_at_s == 8 * 3600
    assert incident.last_seen_s == 13 * 3600
    assert {entry["asset_id"] for entry in incident.candidates} == set(ASSETS[:4])


def test_min_assets_suppresses_an_isolated_spike():
    """
    A single rogue sensor must not open an incident. A real fault moves
    much of the network at once.
    """

    offsets = {ASSETS[0]: {8 * 3600: 25.0}}

    report = _detector(min_assets=3).detect(_frame(offsets))

    assert report.events, "the point detection itself should still be recorded"
    assert not report.detected, "but it must not become an incident"


def test_min_assets_of_one_admits_the_isolated_spike():
    offsets = {ASSETS[0]: {8 * 3600: 25.0}}

    report = _detector(min_assets=1).detect(_frame(offsets))

    assert report.detected


def test_score_is_measured_in_noise_standard_deviations():
    offsets = {asset: {8 * 3600: 5.0} for asset in ASSETS[:3]}

    report = _detector().detect(_frame(offsets))

    # 5 m deviation against a 1 m sensor sigma.
    assert report.events[0].score == pytest.approx(5.0)
    assert report.events[0].residual == pytest.approx(5.0)
    assert report.events[0].expected == pytest.approx(50.0)


def test_separate_windows_stay_separate_incidents():
    offsets = {
        asset: {4 * 3600: 10.0, 5 * 3600: 10.0, 20 * 3600: 10.0}
        for asset in ASSETS[:3]
    }

    report = _detector().detect(_frame(offsets))

    assert len(report.incidents) == 2
    assert [incident.detected_at_s for incident in report.incidents] == [
        4 * 3600,
        20 * 3600,
    ]


def test_candidates_are_ranked_by_severity():
    offsets = {
        ASSETS[0]: {8 * 3600: 6.0},
        ASSETS[1]: {8 * 3600: 20.0},
        ASSETS[2]: {8 * 3600: 12.0},
    }

    report = _detector().detect(_frame(offsets))

    ranked = [entry["asset_id"] for entry in report.incidents[0].candidates]

    assert ranked == [ASSETS[1], ASSETS[2], ASSETS[0]]
    assert report.incidents[0].epicenter == ASSETS[1]


def _mixed_frame(pressure_offsets=None, flow_offsets=None):
    """
    A node and a link that share the ID "101", plus filler assets.

    EPANET namespaces nodes and links separately, so Net3 really does
    contain both a junction 101 and a pipe 101.
    """

    pressure_offsets = pressure_offsets or {}
    flow_offsets = flow_offsets or {}

    rows = []

    for timestamp in TIMESTAMPS:
        for asset in ["101", "102", "103"]:
            rows.append(
                {
                    "timestamp_s": timestamp,
                    "asset_id": asset,
                    "asset_type": "node",
                    "parameter": "pressure",
                    "value": 50.0
                    + pressure_offsets.get(asset, {}).get(timestamp, 0.0),
                    "scenario": "mixed",
                }
            )

        for asset in ["101", "102", "103"]:
            rows.append(
                {
                    "timestamp_s": timestamp,
                    "asset_id": asset,
                    "asset_type": "link",
                    "parameter": "flowrate",
                    "value": 0.5
                    + flow_offsets.get(asset, {}).get(timestamp, 0.0),
                    "scenario": "mixed",
                }
            )

    return pd.DataFrame(rows)


MIXED_SENSORS = SensorModel(
    specs={
        "pressure": SensorSpec(absolute=1.0),
        "flowrate": SensorSpec(absolute=0.001),
    },
    seed=0,
    noise_enabled=False,
)


def _mixed_detector(**kwargs):
    options = {
        "sensor_model": MIXED_SENSORS,
        "score_threshold": 4.0,
        "min_assets": 3,
    }
    options.update(kwargs)

    return ResidualDetector(baseline=_mixed_frame(), **options)


def test_a_node_and_a_link_sharing_an_id_are_kept_apart():
    """
    Keying candidates by asset_id alone let the louder channel delete the
    quieter one, which can remove the faulted asset from the list
    entirely.
    """

    window = [hour * 3600 for hour in range(8, 12)]

    report = _mixed_detector().detect(
        _mixed_frame(
            pressure_offsets={
                asset: {t: 6.0 for t in window} for asset in ["101", "102"]
            },
            flow_offsets={
                asset: {t: 0.02 for t in window} for asset in ["101", "102"]
            },
        )
    )

    candidates = report.incidents[0].candidates

    identities = {
        (entry["asset_id"], entry["parameter"]) for entry in candidates
    }

    assert ("101", "pressure") in identities
    assert ("101", "flowrate") in identities


def test_a_precise_channel_does_not_crowd_out_a_coarse_one():
    """
    The flow channel's noise floor is 1000x finer here, so on raw score
    every flow link outranks every pressure node. A leak is a *node*
    fault that only pressure can name, so interleaving must give the
    pressure channel a seat at the top.
    """

    window = [hour * 3600 for hour in range(8, 12)]

    report = _mixed_detector().detect(
        _mixed_frame(
            # 6 m against 1 m sigma -> score 6
            pressure_offsets={"101": {t: 6.0 for t in window}},
            # 0.02 against 0.001 sigma -> score 20
            flow_offsets={
                asset: {t: 0.02 for t in window}
                for asset in ["101", "102", "103"]
            },
        )
    )

    candidates = report.incidents[0].candidates

    assert candidates[0]["score"] > candidates[1]["score"], (
        "the strongest channel still leads"
    )

    top_two = {entry["parameter"] for entry in candidates[:2]}

    assert top_two == {"pressure", "flowrate"}, (
        f"both channels must appear in the top two, got {top_two}"
    )

    # The pressure node is the best pressure candidate, so it must be the
    # first pressure entry in the merged list.
    pressure_entries = [
        entry for entry in candidates if entry["parameter"] == "pressure"
    ]

    assert pressure_entries[0]["asset_id"] == "101"
    assert pressure_entries[0]["channel_rank"] == 1


def test_min_residual_floor_suppresses_a_trivial_deviation():
    """
    A very precise sensor makes a hydraulically meaningless wobble look
    statistically huge; the floor is what stops that becoming an alarm.
    """

    precise = SensorModel(
        specs={"pressure": SensorSpec(absolute=0.001)},
        seed=0,
        noise_enabled=False,
    )

    offsets = {asset: {8 * 3600: 0.02} for asset in ASSETS[:5]}

    unfloored = _detector(sensor_model=precise).detect(_frame(offsets))
    assert unfloored.detected

    floored = _detector(
        sensor_model=precise,
        min_residual={"pressure": 0.1},
    ).detect(_frame(offsets))

    assert not floored.detected


def test_noise_alone_does_not_open_an_incident():
    """The negative control, in miniature."""

    noisy = SensorModel(
        specs={"pressure": SensorSpec(absolute=1.0)},
        seed=11,
        noise_enabled=True,
    )

    report = _detector(sensor_model=noisy).detect(_frame())

    assert not report.detected


def test_string_and_integer_asset_ids_still_join():
    baseline = _frame()
    baseline["asset_id"] = baseline["asset_id"].astype(int)

    detector = ResidualDetector(
        baseline=baseline,
        sensor_model=QUIET_SENSORS,
    )

    offsets = {asset: {8 * 3600: 10.0} for asset in ASSETS[:3]}

    assert detector.detect(_frame(offsets)).detected


def test_missing_parameter_is_an_explicit_error():
    frame = _frame()
    frame["parameter"] = "temperature"

    with pytest.raises(ValueError, match="No readings for parameters"):
        _detector().detect(frame)


def test_demand_is_excluded_by_default():
    """
    A leak *is* an added demand, so the demand channel echoes the
    injected fault back. Detecting it there would be circular.
    """

    assert "demand" not in _detector().parameters
