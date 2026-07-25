import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from twinrag.simulation.network_loader import WaterNetworkLoader
from twinrag.simulation.simulator import HydraulicSimulator


def main():
    network_path = PROJECT_ROOT / "data" / "networks" / "Net3.inp"

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

    print("\nFirst 5 pressure records:")
    print(pressure.head())

    print("\nFirst 5 demand records:")
    print(demand.head())

    print("\nFirst 5 flow-rate records:")
    print(flowrate.head())


if __name__ == "__main__":
    main()