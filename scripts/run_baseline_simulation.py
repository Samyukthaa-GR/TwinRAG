import sys
from pathlib import Path


# --------------------------------------------------
# Project path setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# --------------------------------------------------
# TwinRAG imports
# These must come AFTER src is added to sys.path
# --------------------------------------------------

from twinrag.simulation.network_loader import WaterNetworkLoader
from twinrag.simulation.simulator import HydraulicSimulator
from twinrag.simulation.result_formatter import SimulationResultFormatter


def main():
    network_path = (
        PROJECT_ROOT
        / "data"
        / "networks"
        / "Net3.inp"
    )

    print("Loading EPANET network...")

    loader = WaterNetworkLoader(network_path)
    network = loader.load()

    print("Configuring 24-hour simulation...")

    simulator = HydraulicSimulator(network)

    simulator.configure_simulation(
        duration_hours=24,
        hydraulic_timestep_seconds=3600,
        report_timestep_seconds=3600,
    )

    print("Running hydraulic simulation...")

    simulator.run()

    pressure = simulator.get_pressure()
    demand = simulator.get_demand()
    flowrate = simulator.get_flowrate()

    print("\nSimulation completed successfully.")

    print("\nPressure data shape:")
    print(pressure.shape)

    print("\nDemand data shape:")
    print(demand.shape)

    print("\nFlow-rate data shape:")
    print(flowrate.shape)

    # --------------------------------------------------
    # Format baseline data
    # --------------------------------------------------

    formatter = SimulationResultFormatter(
        scenario_name="normal"
    )

    baseline_data = formatter.combine(
        pressure=pressure,
        demand=demand,
        flowrate=flowrate,
    )

    # --------------------------------------------------
    # Save baseline dataset
    # --------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "baseline.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_data.to_csv(
        output_path,
        index=False,
    )

    print("\nBaseline dataset created successfully.")
    print(f"Saved to: {output_path}")

    print("\nBaseline dataset shape:")
    print(baseline_data.shape)

    print("\nFirst 10 baseline records:")
    print(baseline_data.head(10))


if __name__ == "__main__":
    main()