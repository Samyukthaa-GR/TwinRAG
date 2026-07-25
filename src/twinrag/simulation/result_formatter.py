import pandas as pd


class SimulationResultFormatter:
    """
    Converts raw WNTR simulation results into standardized
    long-format datasets.
    """

    def __init__(self, scenario_name: str = "normal"):
        self.scenario_name = scenario_name

    def format_node_parameter(
        self,
        dataframe: pd.DataFrame,
        parameter: str,
        asset_type: str = "node",
    ) -> pd.DataFrame:
        """
        Convert a node-based WNTR dataframe from wide to long format.
        """

        formatted = (
            dataframe
            .reset_index()
            .melt(
                id_vars="index",
                var_name="asset_id",
                value_name="value",
            )
        )

        formatted = formatted.rename(
            columns={"index": "timestamp_s"}
        )

        formatted["asset_type"] = asset_type
        formatted["parameter"] = parameter
        formatted["scenario"] = self.scenario_name

        return formatted[
            [
                "timestamp_s",
                "asset_id",
                "asset_type",
                "parameter",
                "value",
                "scenario",
            ]
        ]

    def format_link_parameter(
        self,
        dataframe: pd.DataFrame,
        parameter: str,
        asset_type: str = "link",
    ) -> pd.DataFrame:
        """
        Convert a link-based WNTR dataframe from wide to long format.
        """

        return self.format_node_parameter(
            dataframe=dataframe,
            parameter=parameter,
            asset_type=asset_type,
        )

    def combine(
        self,
        pressure: pd.DataFrame,
        demand: pd.DataFrame,
        flowrate: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine pressure, demand and flow-rate data into
        one standardized dataset.
        """

        pressure_long = self.format_node_parameter(
            pressure,
            parameter="pressure",
            asset_type="node",
        )

        demand_long = self.format_node_parameter(
            demand,
            parameter="demand",
            asset_type="node",
        )

        flow_long = self.format_link_parameter(
            flowrate,
            parameter="flowrate",
            asset_type="link",
        )

        combined = pd.concat(
            [
                pressure_long,
                demand_long,
                flow_long,
            ],
            ignore_index=True,
        )

        return combined