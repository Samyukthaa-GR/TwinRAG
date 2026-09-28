"""
Shared detector machinery.

``AnomalyDetector`` fixes the interface (``detect(dataset) -> AnomalyReport``)
and owns the step every detector needs: rolling thousands of individual
point detections up into a small number of incidents.

That roll-up is where the false-alarm rate is actually controlled. At a
score threshold of 4 sigma, a scenario's ~5000 readings will throw a
fraction of a spurious point detection by chance. But a genuine fault
moves 55-95 of Net3's 97 nodes within one timestep, so requiring several
assets to disagree simultaneously separates physics from instrument
noise far more cleanly than raising the per-reading threshold would --
and it does so without blunting sensitivity to weak faults.
"""

from abc import ABC, abstractmethod

from .incidents import AnomalyEvent, AnomalyReport, Incident


class AnomalyDetector(ABC):
    """
    Base class for anomaly detectors.
    """

    name: str = "detector"

    #: Parameters a detector looks at unless told otherwise.
    #:
    #: ``demand`` is excluded deliberately. ``LeakFault`` *is* an added
    #: demand at the target junction, so the demand channel reports the
    #: injected fault back verbatim -- the residual there equals
    #: ``severity x max_leak_demand`` to the last decimal. Detecting a
    #: leak that way is circular, and no real network meters demand at
    #: every junction anyway.
    default_parameters = ("pressure", "flowrate")

    def __init__(
        self,
        parameters=None,
        score_threshold: float = 4.0,
        min_assets: int = 3,
    ):
        """
        Parameters
        ----------
        parameters
            Measured parameters to examine.
        score_threshold
            Flag a reading when its normalised deviation exceeds this.
            In units of standard deviations.
        min_assets
            How many distinct assets must be flagged at the same
            timestamp before that timestamp counts as an incident.
        """

        self.parameters = tuple(parameters or self.default_parameters)

        if score_threshold <= 0:
            raise ValueError("score_threshold must be positive.")

        if min_assets < 1:
            raise ValueError("min_assets must be at least 1.")

        self.score_threshold = float(score_threshold)
        self.min_assets = int(min_assets)

    @abstractmethod
    def detect(self, dataset) -> AnomalyReport:
        """
        Find anomalies in a long-format dataset.
        """

    def describe(self) -> dict:
        """
        Configuration, recorded into the report for reproducibility.
        """

        return {
            "detector": self.name,
            "parameters": list(self.parameters),
            "score_threshold": self.score_threshold,
            "min_assets": self.min_assets,
        }

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _scenario_name(self, dataset) -> str:
        if "scenario" not in dataset.columns or dataset.empty:
            return ""

        values = dataset["scenario"].dropna().unique()

        return str(values[0]) if len(values) else ""

    def _events_from_flags(
        self,
        frame,
        scenario: str,
    ) -> list:
        """
        Turn flagged rows of a scored frame into ``AnomalyEvent`` objects.

        ``frame`` must carry ``observed``, ``expected``, ``residual`` and
        ``score`` columns alongside the standard schema.
        """

        flagged = frame[frame["score"] > self.score_threshold]

        return [
            AnomalyEvent(
                timestamp_s=int(row.timestamp_s),
                asset_id=str(row.asset_id),
                asset_type=str(row.asset_type),
                parameter=str(row.parameter),
                observed=float(row.observed),
                expected=float(row.expected),
                residual=float(row.residual),
                score=float(row.score),
                scenario=scenario,
            )
            for row in flagged.itertuples(index=False)
        ]

    def _build_incidents(
        self,
        events: list,
        timeline: list,
        scenario: str,
    ) -> list:
        """
        Group point detections into incidents.

        Parameters
        ----------
        events
            Point detections from ``_events_from_flags``.
        timeline
            Every timestamp present in the dataset, ascending. Needed to
            tell "the next report step" from "a gap", so two separate
            faults in one run do not merge into one incident.
        scenario
            Scenario label carried onto each incident.
        """

        if not events:
            return []

        by_timestamp = {}

        for event in events:
            by_timestamp.setdefault(event.timestamp_s, []).append(event)

        # A timestamp only counts if enough distinct assets disagree.
        # Keyed by (type, id) because Net3 contains both a node "101" and
        # a pipe "101" -- they are separate assets that happen to share a
        # name, and counting them as one would understate the spread.
        positive = {
            timestamp
            for timestamp, group in by_timestamp.items()
            if len({(event.asset_type, event.asset_id) for event in group})
            >= self.min_assets
        }

        if not positive:
            return []

        incidents = []
        current = []

        for timestamp in timeline:
            if timestamp in positive:
                current.append(timestamp)
                continue

            if current:
                incidents.append(
                    self._make_incident(current, by_timestamp, scenario)
                )
                current = []

        if current:
            incidents.append(
                self._make_incident(current, by_timestamp, scenario)
            )

        return incidents

    def _make_incident(
        self,
        timestamps: list,
        by_timestamp: dict,
        scenario: str,
    ) -> Incident:
        """
        Collapse one contiguous run of anomalous timestamps.
        """

        members = [
            event
            for timestamp in timestamps
            for event in by_timestamp[timestamp]
        ]

        # Keep each asset's strongest deviation in the window. Peak
        # severity localises better than a mean, which a long tail of
        # mildly-affected neighbours would drag down.
        #
        # Keyed by (parameter, asset_id): a node and a link can share a
        # name in EPANET, and one channel's reading must never overwrite
        # the other's -- that would drop the faulted asset outright.
        #
        # ``first_seen_s`` is when the asset first disagreed; ``peak_at_s``
        # is when the kept (strongest) reading happened. They differ, and
        # the difference matters: a pump outage first shows as the pump
        # delivering nothing, while its peak can come hours later when it
        # restarts and runs *above* normal to refill an empty tank.
        best = {}
        first_seen = {}

        for event in members:
            key = (event.parameter, event.asset_id)

            first_seen[key] = min(first_seen.get(key, event.timestamp_s), event.timestamp_s)

            existing = best.get(key)

            if existing is None or event.score > existing["score"]:
                best[key] = {
                    "asset_id": event.asset_id,
                    "asset_type": event.asset_type,
                    "parameter": event.parameter,
                    "score": event.score,
                    "residual": event.residual,
                    "peak_at_s": event.timestamp_s,
                }

        for key, entry in best.items():
            entry["first_seen_s"] = first_seen[key]

        candidates = self._rank_candidates(list(best.values()))

        return Incident(
            scenario=scenario,
            detected_at_s=min(timestamps),
            last_seen_s=max(timestamps),
            candidates=candidates,
            event_count=len(members),
            parameters=sorted({event.parameter for event in members}),
            detector=self.name,
        )

    def _rank_candidates(self, entries: list) -> list:
        """
        Order candidates so no measurement channel crowds out another.

        Scores are sigma-normalised, which makes them comparable *within*
        a parameter and misleading *across* parameters. A flow meter's
        noise floor is around 0.001 m3/s against a pressure transducer's
        0.30 m, so a hydraulically trivial flow wobble outscores a 3 m
        pressure collapse. Ranking on raw score therefore sorts by sensor
        precision rather than by evidence: in a leak scenario the top 20
        candidates come back as flow links while the leaking junction --
        the *only* asset that can be the answer, since a leak is a node
        fault -- sits far below.

        So each channel is ranked internally, then the channels are
        interleaved: every parameter's strongest candidate appears before
        any parameter's second. Each fault class is named by the channel
        that can actually see it (pressure for leaks, flow for closures),
        and interleaving guarantees both get a seat near the top.

        Ties within a tier fall back to raw score, which is
        parameter-agnostic but only ever breaks ties.
        """

        by_parameter = {}

        for entry in entries:
            by_parameter.setdefault(entry["parameter"], []).append(entry)

        for group in by_parameter.values():
            group.sort(key=lambda entry: entry["score"], reverse=True)

            for rank, entry in enumerate(group, start=1):
                entry["channel_rank"] = rank

        return sorted(
            entries,
            key=lambda entry: (entry["channel_rank"], -entry["score"]),
        )

    def _report(
        self,
        scenario: str,
        events: list,
        timeline: list,
        extra_config: dict | None = None,
    ) -> AnomalyReport:
        config = self.describe()

        if extra_config:
            config.update(extra_config)

        return AnomalyReport(
            scenario=scenario,
            detector=self.name,
            events=events,
            incidents=self._build_incidents(events, timeline, scenario),
            config=config,
        )
