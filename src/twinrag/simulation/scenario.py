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

        config = self.simulation_config

        # Averaged reporting needs every hydraulic step, which are then
        # folded into report-period means in _dataset.
        report_timestep = (
            config.hydraulic_timestep_seconds
            if getattr(config, "report_average", False)
            else config.report_timestep_seconds
        )

        simulator.configure_simulation(
            duration_hours=config.duration_hours,
            hydraulic_timestep_seconds=config.hydraulic_timestep_seconds,
            report_timestep_seconds=report_timestep,
        )

        simulator.run()

        return simulator

    def _report_means(self, frame):
        """
        Fold fine-step results into one row per report period: the value
        stamped at ``t`` is the mean over ``(t - period, t]`` -- what a
        meter logging each interval's average (or totalised volume) would
        record. The first row, ``t = 0``, is the instant itself.

        Instantaneous hourly snapshots miss anything shorter than an hour:
        the building's transfer pump runs ~40 minutes between two
        snapshots, so a pump that silently stopped looked exactly like a
        normal day.
        """

        if not getattr(self.simulation_config, "report_average", False):
            return frame

        period = self.simulation_config.report_timestep_seconds
        index = frame.index.to_numpy()
        # (t - period, t]  ->  bucket end t = ceil(index / period) * period
        buckets = -(-index // period) * period

        return frame.groupby(buckets).mean()

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

        pressure = simulator.get_pressure()

        floor = getattr(
            self.simulation_config,
            "pressure_floor_m",
            None,
        )

        if floor is not None:
            # Sections cut off from every source come back from EPANET
            # with artefact heads; a transducer on a drained pipe reads
            # atmospheric. See SimulationConfig.pressure_floor_m.
            # Clipped before averaging so the artefacts never reach a mean.
            pressure = pressure.clip(lower=floor)

        return formatter.combine(
            pressure=self._report_means(pressure),
            demand=self._report_means(simulator.get_demand()),
            flowrate=self._report_means(simulator.get_flowrate()),
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