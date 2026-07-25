from pathlib import Path

import wntr


class WaterNetworkLoader:
    """
    Loads and validates an EPANET water distribution network.

    This class acts as the entry point for the Digital Twin simulation module.
    """

    def __init__(self, inp_path: str):
        self.inp_path = Path(inp_path)
        self.network = None

    def validate_file(self) -> None:
        """
        Validate that the EPANET input file exists and has the correct extension.
        """

        if not self.inp_path.exists():
            raise FileNotFoundError(
                f"EPANET network file not found: {self.inp_path}"
            )

        if self.inp_path.suffix.lower() != ".inp":
            raise ValueError(
                f"Expected an EPANET '.inp' file, got: {self.inp_path.suffix}"
            )

    def load(self):
        """
        Load the EPANET network using WNTR.
        """

        self.validate_file()

        try:
            self.network = wntr.network.WaterNetworkModel(
                str(self.inp_path)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load EPANET network: {self.inp_path}"
            ) from exc

        return self.network

    def get_summary(self) -> dict:
        """
        Return a structured summary of the loaded network.
        """

        if self.network is None:
            raise RuntimeError(
                "Network has not been loaded yet. Call load() first."
            )

        wn = self.network

        summary = {
            "network_file": self.inp_path.name,
            "junctions": wn.num_junctions,
            "reservoirs": wn.num_reservoirs,
            "tanks": wn.num_tanks,
            "pipes": wn.num_pipes,
            "pumps": wn.num_pumps,
            "valves": wn.num_valves,
            "nodes_total": wn.num_nodes,
            "links_total": wn.num_links,
        }

        return summary

    def print_summary(self) -> None:
        """
        Print the network summary in a readable format.
        """

        summary = self.get_summary()

        print("\n" + "=" * 50)
        print("WATER DISTRIBUTION NETWORK SUMMARY")
        print("=" * 50)

        print(f"Network file : {summary['network_file']}")

        print("\nNodes")
        print("-" * 50)
        print(f"Junctions    : {summary['junctions']}")
        print(f"Reservoirs   : {summary['reservoirs']}")
        print(f"Tanks        : {summary['tanks']}")
        print(f"Total nodes  : {summary['nodes_total']}")

        print("\nLinks")
        print("-" * 50)
        print(f"Pipes        : {summary['pipes']}")
        print(f"Pumps        : {summary['pumps']}")
        print(f"Valves       : {summary['valves']}")
        print(f"Total links  : {summary['links_total']}")

        print("=" * 50)