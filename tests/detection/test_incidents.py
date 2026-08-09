from twinrag.detection import AnomalyEvent, AnomalyReport, Incident


def _event(timestamp_s=3600, asset_id="101", score=5.0):
    return AnomalyEvent(
        timestamp_s=timestamp_s,
        asset_id=asset_id,
        asset_type="node",
        parameter="pressure",
        observed=42.0,
        expected=50.0,
        residual=-8.0,
        score=score,
        scenario="test",
    )


DEFAULT_CANDIDATES = [
    {"asset_id": "101", "asset_type": "node", "score": 9.0},
    {"asset_id": "105", "asset_type": "node", "score": 4.0},
]


def _incident(detected_at_s=3600, candidates=None):
    return Incident(
        scenario="test",
        detected_at_s=detected_at_s,
        last_seen_s=detected_at_s + 3600,
        candidates=(
            DEFAULT_CANDIDATES if candidates is None else candidates
        ),
        event_count=2,
        parameters=["pressure"],
        detector="residual",
    )


def test_epicenter_is_the_top_candidate():
    assert _incident().epicenter == "101"


def test_epicenter_of_an_empty_candidate_list_is_none():
    assert _incident(candidates=[]).epicenter is None


def test_top_candidates_respects_the_limit():
    assert len(_incident().top_candidates(limit=1)) == 1


def test_first_detection_is_the_earliest_incident():
    report = AnomalyReport(
        scenario="test",
        detector="residual",
        incidents=[_incident(7200), _incident(3600)],
    )

    assert report.first_detection_s == 3600
    assert report.detected


def test_an_empty_report_has_no_first_detection():
    report = AnomalyReport(scenario="test", detector="residual")

    assert report.first_detection_s is None
    assert not report.detected


def test_events_dataframe_orders_by_time_then_severity():
    report = AnomalyReport(
        scenario="test",
        detector="residual",
        events=[
            _event(timestamp_s=7200, score=3.0),
            _event(timestamp_s=3600, score=4.0),
            _event(timestamp_s=3600, asset_id="105", score=9.0),
        ],
    )

    frame = report.events_dataframe()

    assert frame["timestamp_s"].tolist() == [3600, 3600, 7200]
    assert frame["score"].tolist() == [9.0, 4.0, 3.0]


def test_empty_events_dataframe_keeps_the_schema():
    """Downstream code should not have to special-case a quiet scenario."""

    frame = AnomalyReport(scenario="test", detector="residual").events_dataframe()

    assert frame.empty
    assert "asset_id" in frame.columns
    assert "score" in frame.columns


def test_report_serialises_for_the_retrieval_layer():
    report = AnomalyReport(
        scenario="test",
        detector="residual",
        events=[_event()],
        incidents=[_incident()],
        config={"score_threshold": 4.0},
    )

    payload = report.to_dict()

    assert payload["detected"] is True
    assert payload["event_count"] == 1
    assert payload["incidents"][0]["epicenter"] == "101"
    assert payload["config"]["score_threshold"] == 4.0
