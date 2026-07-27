"""
Scenario orchestration.

``FaultScenarioRunner`` ties the simulation stages together for a
single scenario:

    load network -> optionally inject fault -> simulate ->
    format dataset -> add temporal state labels ->
    return dataset + ground-truth metadata

A fresh network is loaded for every scenario so mutations from one
fault can never leak into another scenario.
"""

from pathlib import Path

from .network_loader import WaterNetworkLoader
from .simulator import HydraulicSimulator
from .result_formatter import SimulationResultFormatter


class FaultScenarioRunner:
    """
    Coordinates baseline and fault-condition hydraulic simulations.
    """

    def __init__(
        self,
        network_path,
        simulation_config,
    ):
        self.network_path = Path(network_path)
        self.simulation_config = simulation_config

    def _load_network(self):
        """
        Load a fresh copy of the EPANET network.
        """

        return WaterNetworkLoader(
            self.network_path
        ).load()

    def _simulate(
        self,
        network,
    ) -> HydraulicSimulator:
        """
        Configure and run the hydraulic simulation.
        """

        simulator = HydraulicSimulator(
            network
        )

        simulator.configure_simulation(
            duration_hours=(
                self.simulation_config.duration_hours
            ),
            hydraulic_timestep_seconds=(
                self.simulation_config.hydraulic_timestep_seconds
            ),
            report_timestep_seconds=(
                self.simulation_config.report_timestep_seconds
            ),
        )

        simulator.run()

        return simulator

    def _dataset(
        self,
        simulator,
        scenario_name,
    ):
        """
        Convert raw WNTR outputs into the standard long-form dataset.
        """

        formatter = SimulationResultFormatter(
            scenario_name=scenario_name
        )

        return formatter.combine(
            pressure=simulator.get_pressure(),
            demand=simulator.get_demand(),
            flowrate=simulator.get_flowrate(),
        )

    def _add_baseline_state_labels(
        self,
        dataset,
    ):
        """
        Label every baseline timestamp as normal.
        """

        dataset = dataset.copy()

        dataset["state"] = "normal"

        return dataset

    def _add_fault_state_labels(
        self,
        dataset,
        fault,
    ):
        """
        Add temporal operating-state labels to a fault scenario.

        States
        ------
        normal:
            Time before the fault begins.

        fault_active:
            Time during which the injected fault is active.

        recovery:
            Time after the fault has been removed. The network may
            still differ from baseline because hydraulic state has
            evolved during the fault.
        """

        dataset = dataset.copy()

        start_time_s = (
            fault.start_hour * 3600
        )

        end_time_s = (
            fault.end_hour * 3600
            if fault.end_hour is not None
            else None
        )

        # Everything starts as normal.
        dataset["state"] = "normal"

        if end_time_s is None:
            # Fault remains active until simulation ends.
            dataset.loc[
                dataset["timestamp_s"] >= start_time_s,
                "state",
            ] = "fault_active"

        else:
            # Fault interval:
            # start_time <= t < end_time
            dataset.loc[
                (
                    dataset["timestamp_s"]
                    >= start_time_s
                )
                & (
                    dataset["timestamp_s"]
                    < end_time_s
                ),
                "state",
            ] = "fault_active"

            # Anything after fault removal is recovery.
            dataset.loc[
                dataset["timestamp_s"]
                >= end_time_s,
                "state",
            ] = "recovery"

        return dataset

    def run_baseline(self):
        """
        Simulate the unmodified water network.

        Returns
        -------
        dataset, metadata
        """

        network = self._load_network()

        simulator = self._simulate(
            network
        )

        dataset = self._dataset(
            simulator,
            scenario_name="normal",
        )

        dataset = (
            self._add_baseline_state_labels(
                dataset
            )
        )

        metadata = {
            "scenario": "normal",
            "fault_type": None,
            "network_file": (
                self.network_path.name
            ),
            "duration_hours": (
                self.simulation_config.duration_hours
            ),
        }

        return dataset, metadata

    def run_fault(
        self,
        fault,
    ):
        """
        Inject a fault into a fresh network, simulate it,
        and return labelled hydraulic data plus ground truth.
        """

        network = self._load_network()

        fault.apply(
            network
        )

        simulator = self._simulate(
            network
        )

        dataset = self._dataset(
            simulator,
            scenario_name=(
                fault.scenario_name()
            ),
        )

        dataset = (
            self._add_fault_state_labels(
                dataset,
                fault,
            )
        )

        metadata = fault.describe()

        metadata.update(
            {
                "network_file": (
                    self.network_path.name
                ),
                "duration_hours": (
                    self.simulation_config.duration_hours
                ),
            }
        )

        return dataset, metadata