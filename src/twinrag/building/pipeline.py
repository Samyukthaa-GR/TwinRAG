"""
The building's end-to-end diagnosis pipeline, in one place:

    hydraulic dataset
      -> ResidualDetector (building sensors + instrument noise)   Phase 3
      -> SubgraphRetriever over the BuildingKnowledgeGraph         Phase 4
      -> RuleBasedDiagnoser (alert message)                        Phase 5

Used by the 3D viewer build and available to any other consumer, so the
dashboard shows exactly what the evaluated pipeline produces. The answer
key never enters: only the dataset and the twin's baseline do.

``run`` diagnoses a whole day with hindsight (what the evaluation
scores). ``run_live`` replays the day hour by hour, using only the
readings up to each hour -- what an operator would actually have seen.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinrag.detection import ResidualDetector, SensorModel
from twinrag.graph.knowledge import BuildingKnowledgeGraph
from twinrag.reasoning.rules import RuleBasedDiagnoser
from twinrag.retrieval import RetrievalConfig, SubgraphRetriever

from .sensors import building_sensor_model


@dataclass
class PipelineConfig:
    threshold: float = 4.0       # sigma of instrument noise
    min_assets: int = 2          # sensors that must disagree at once
    hops: int = 2


class BuildingDiagnosisPipeline:
    """
    Detect, retrieve and diagnose every incident in a building dataset.
    """

    def __init__(self, kg: BuildingKnowledgeGraph, baseline, config: PipelineConfig | None = None):
        self.kg = kg
        self.baseline = baseline
        self.config = config or PipelineConfig()
        self.retriever = SubgraphRetriever(kg, RetrievalConfig(hops=self.config.hops))
        self.diagnoser = RuleBasedDiagnoser()

    def _diagnose(self, dataset, sensor_model) -> list:
        detector = ResidualDetector(
            self.baseline,
            sensor_model=sensor_model,
            score_threshold=self.config.threshold,
            min_assets=self.config.min_assets,
        )
        report = detector.detect(dataset)

        out = []
        for number, incident in enumerate(report.incidents, start=1):
            packet = self.retriever.retrieve(
                incident, incident_id=f"INC-{number}", events=report.events
            )
            result = self.diagnoser.diagnose(packet.to_dict())
            out.append(
                {
                    "id": f"INC-{number}",
                    "first_alarm_hour": incident.detected_at_s // 3600,
                    "last_alarm_hour": incident.last_seen_s // 3600,
                    "packet": packet.to_dict(),
                    "diagnosis": result.diagnosis,
                    "grounded": result.grounded,
                }
            )
        return out

    def run(self, dataset, seed: int = 0) -> list:
        """
        Every incident of the day, diagnosed with the whole day's data.

        Returns one dict per detected incident, in time order::

            {"id", "first_alarm_hour", "last_alarm_hour",
             "packet": EvidencePacket dict, "diagnosis": dict | None,
             "grounded": bool}
        """

        return self._diagnose(dataset, building_sensor_model(self.kg.layout, seed=seed))

    def run_live(self, dataset, seed: int = 0) -> dict:
        """
        What the system would have reported at each hour, from the readings
        available by then.

        Sensor noise is drawn once for the whole day and the truncated
        frames reuse it: re-drawing per hour would change a past reading's
        noise as later hours arrive, and the diagnosis would flicker.

        Returns ``{hour: [incident, ...]}`` for every hour from the first
        detection onward (incidents as in ``run``, as known at that hour).
        """

        if not self.run(dataset, seed):
            return {}

        noisy = building_sensor_model(self.kg.layout, seed=seed).observe(dataset)
        frame = noisy.drop(columns=["truth", "noise_std"])

        # Already noisy: pass readings through, but keep coverage and the
        # instrument noise scale the detector divides by.
        quiet = building_sensor_model(self.kg.layout, seed=seed, noise_enabled=False)

        hours = sorted(int(t) // 3600 for t in frame["timestamp_s"].unique())
        live = {}
        for hour in hours:
            part = frame[frame["timestamp_s"] <= hour * 3600]
            incidents = self._diagnose(part, quiet)
            if incidents:
                live[hour] = incidents

        return live
