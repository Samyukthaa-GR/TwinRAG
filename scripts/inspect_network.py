import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from twinrag.simulation.network_loader import WaterNetworkLoader


def main():
    network_path = PROJECT_ROOT / "data" / "networks" / "Net3.inp"

    loader = WaterNetworkLoader(network_path)

    loader.load()

    loader.print_summary()


if __name__ == "__main__":
    main()