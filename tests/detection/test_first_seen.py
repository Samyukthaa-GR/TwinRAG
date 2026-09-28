"""
An incident candidate records both when its asset first disagreed with
the twin and when its strongest reading happened -- they differ, and the
retrieval layer needs the true first time to order cause and effect.
"""

from twinrag.detection import AnomalyEvent, StatisticalDetector


def _event(hour, residual):
    return AnomalyEvent(timestamp_s=hour * 3600, asset_id="PUMP1", asset_type="link",
                        parameter="flowrate", observed=0.0, expected=0.0,
                        residual=residual, score=abs(residual) * 1000)


def test_first_seen_is_the_first_disagreement_not_the_peak():
    events = {11: [_event(11, -2.5)], 12: [_event(12, -2.5)], 16: [_event(16, 2.8)]}

    incident = StatisticalDetector()._make_incident([11, 12, 16], events, "s")
    candidate = incident.candidates[0]

    assert candidate["first_seen_s"] == 11 * 3600
    assert candidate["peak_at_s"] == 16 * 3600
    assert candidate["residual"] == 2.8
