from pathlib import Path

import pandas as pd
import wntr


class HydraulicSimulator:
    """
    Runs hydraulic simulations for a WNTR water network model.
    """

    def __init__(self, network):
        self.network = network
        self.results = None

    def configure_simulation(
        self,
        duration_hours: int = 24,
        hydraulic_timestep_seconds: int = 3600,
        report_timestep_seconds: int = 3600,
    ) -> None:
        """
        Configure the simulation duration and time steps.
        """

        self.network.options.time.duration = duration_hours * 3600
        self.network.options.time.hydraulic_timestep = hydraulic_timestep_seconds
        self.network.options.time.report_timestep = report_timestep_seconds

    def run(self):
        """
        Run the hydraulic simulation using EPANET.
        """

        simulator = wntr.sim.EpanetSimulator(self.network)

        self.results = simulator.run_sim()

        return self.results

    def get_pressure(self) -> pd.DataFrame:
        """
        Return node pressure time-series data.
        """

        self._ensure_results_exist()

        return self.results.node["pressure"].copy()

    def get_demand(self) -> pd.DataFrame:
        """
        Return node demand time-series data.
        """

        self._ensure_results_exist()

        return self.results.node["demand"].copy()

    def get_flowrate(self) -> pd.DataFrame:
        """
        Return link flow-rate time-series data.
        """

        self._ensure_results_exist()

        return self.results.link["flowrate"].copy()

    def _ensure_results_exist(self) -> None:
        """
        Ensure that a simulation has been run before accessing results.
        """

        if self.results is None:
            raise RuntimeError(
                "Simulation results are not available. Run the simulation first."
            )