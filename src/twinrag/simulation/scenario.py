"""
Scenario orchestration.

``FaultScenarioRunner`` ties the existing simulation stages together for a
single scenario:

    load network -> (optionally inject fault) -> simulate ->
    format to the long dataset -> return (dataset, ground-truth metadata)

A **fresh** network is loaded for every scenario so a mutation from one fault
can never leak into the next.
"""

from pathlib import Path

from .network_loader import WaterNetworkLoader
from .simulator import HydraulicSimulator
from .result_formatter import SimulationResultFormatter


class FaultScenarioRunner:
    def __init__(self, network_path, simulation_config):
        self.network_path = Path(network_path)
        self.simulation_config = simulation_config

    def _load_network(self):
        return WaterNetworkLoader(self.network_path).load()

    def _simulate(self, network) -> HydraulicSimulator:
        simulator = HydraulicSimulator(network)
        simulator.configure_simulation(
            duration_hours=self.simulation_config.duration_hours,
            hydraulic_timestep_seconds=self.simulation_config.hydraulic_timestep_seconds,
            report_timestep_seconds=self.simulation_config.report_timestep_seconds,
        )
        simulator.run()
        return simulator

    def _dataset(self, simulator, scenario_name):
        formatter = SimulationResultFormatter(scenario_name=scenario_name)
        return formatter.combine(
            pressure=simulator.get_pressure(),
            demand=simulator.get_demand(),
            flowrate=simulator.get_flowrate(),
        )

    def run_baseline(self):
        """Simulate the unmodified network. Returns ``(dataset, metadata)``."""

        network = self._load_network()
        simulator = self._simulate(network)
        dataset = self._dataset(simulator, scenario_name="normal")

        metadata = {
            "scenario": "normal",
            "fault_type": None,
            "network_file": self.network_path.name,
            "duration_hours": self.simulation_config.duration_hours,
        }
        return dataset, metadata

    def run_fault(self, fault):
        """
        Inject ``fault`` into a fresh network, simulate, and return
        ``(dataset, metadata)`` where ``metadata`` is the fault's ground truth.
        """

        network = self._load_network()
        fault.apply(network)

        simulator = self._simulate(network)
        dataset = self._dataset(simulator, scenario_name=fault.scenario_name())

        metadata = fault.describe()
        metadata.update(
            {
                "network_file": self.network_path.name,
                "duration_hours": self.simulation_config.duration_hours,
            }
        )
        return dataset, metadata
