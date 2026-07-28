"""Dataset loading and validation utilities for Module 2.

This module loads the normal Digital Twin baseline and fault-scenario
datasets produced by Module 1. It validates their structure before they
are passed to the alignment and residual-calculation stages.
"""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp_s",
    "asset_id",
    "asset_type",
    "parameter",
    "value",
    "scenario",
    "state",
}

ALIGNMENT_COLUMNS = [
    "timestamp_s",
    "asset_id",
    "asset_type",
    "parameter",
]

SUPPORTED_PARAMETERS = {
    "pressure",
    "demand",
    "flowrate",
}


class DetectionDataError(ValueError):
    """Raised when a Module 2 input dataset is invalid."""


def _validate_file_path(file_path: str | Path) -> Path:
    """Validate that the supplied dataset path exists and is a CSV file.

    Args:
        file_path: Path to the dataset.

    Returns:
        Validated Path object.

    Raises:
        FileNotFoundError: If the path does not exist.
        DetectionDataError: If the path is not a CSV file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    if not path.is_file():
        raise DetectionDataError(f"Dataset path is not a file: {path}")

    if path.suffix.lower() != ".csv":
        raise DetectionDataError(
            f"Expected a CSV dataset, but received: {path}"
        )

    return path


def _validate_required_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Verify that the dataset contains all required Module 1 columns."""
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise DetectionDataError(
            f"{dataset_name} is missing required columns: {missing_text}"
        )


def _validate_empty_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Reject empty datasets."""
    if dataframe.empty:
        raise DetectionDataError(f"{dataset_name} contains no rows.")


def _validate_missing_values(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Verify that required columns contain no missing values."""
    missing_counts = dataframe[list(REQUIRED_COLUMNS)].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        details = ", ".join(
            f"{column}={count}"
            for column, count in missing_counts.items()
        )
        raise DetectionDataError(
            f"{dataset_name} contains missing values: {details}"
        )


def _validate_duplicate_alignment_keys(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Verify that each timestamp-asset-parameter record is unique."""
    duplicate_mask = dataframe.duplicated(
        subset=ALIGNMENT_COLUMNS,
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())

        example_duplicates = (
            dataframe.loc[duplicate_mask, ALIGNMENT_COLUMNS]
            .head(5)
            .to_dict(orient="records")
        )

        raise DetectionDataError(
            f"{dataset_name} contains {duplicate_count} rows with duplicate "
            f"alignment keys. Examples: {example_duplicates}"
        )


def _validate_parameters(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Verify that the dataset contains only supported hydraulic parameters."""
    parameters = set(dataframe["parameter"].astype(str).unique())
    unsupported = parameters.difference(SUPPORTED_PARAMETERS)

    if unsupported:
        unsupported_text = ", ".join(sorted(unsupported))
        raise DetectionDataError(
            f"{dataset_name} contains unsupported parameters: "
            f"{unsupported_text}"
        )


def validate_detection_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Run all common validation checks for a detection input dataset.

    Args:
        dataframe: Dataset to validate.
        dataset_name: Human-readable name used in error messages.

    Raises:
        DetectionDataError: If the dataset is invalid.
    """
    _validate_empty_dataset(dataframe, dataset_name)
    _validate_required_columns(dataframe, dataset_name)
    _validate_missing_values(dataframe, dataset_name)
    _validate_duplicate_alignment_keys(dataframe, dataset_name)
    _validate_parameters(dataframe, dataset_name)


def load_baseline_dataset(
    file_path: str | Path,
) -> pd.DataFrame:
    """Load and validate the normal Digital Twin baseline dataset.

    The baseline must contain exactly one scenario named ``normal`` and
    must contain only rows whose state is ``normal``.

    Args:
        file_path: Path to the baseline CSV.

    Returns:
        Validated baseline DataFrame.

    Raises:
        DetectionDataError: If the baseline metadata is invalid.
    """
    path = _validate_file_path(file_path)
    dataframe = pd.read_csv(path, dtype={"asset_id": str})

    validate_detection_dataset(
        dataframe=dataframe,
        dataset_name="Baseline dataset",
    )

    scenarios = set(dataframe["scenario"].astype(str).unique())

    if scenarios != {"normal"}:
        raise DetectionDataError(
            "Baseline dataset must contain exactly one scenario named "
            f"'normal'. Found: {sorted(scenarios)}"
        )

    states = set(dataframe["state"].astype(str).unique())

    if states != {"normal"}:
        raise DetectionDataError(
            "Baseline dataset must contain only the 'normal' state. "
            f"Found: {sorted(states)}"
        )

    return dataframe


def load_scenario_dataset(
    file_path: str | Path,
) -> pd.DataFrame:
    """Load and validate one generated fault-scenario dataset.

    Args:
        file_path: Path to a generated scenario CSV.

    Returns:
        Validated fault-scenario DataFrame.

    Raises:
        DetectionDataError: If multiple scenario names are present.
    """
    path = _validate_file_path(file_path)
    dataframe = pd.read_csv(path, dtype={"asset_id": str})

    validate_detection_dataset(
        dataframe=dataframe,
        dataset_name="Scenario dataset",
    )

    scenarios = dataframe["scenario"].astype(str).unique()

    if len(scenarios) != 1:
        raise DetectionDataError(
            "A scenario dataset must contain exactly one scenario name. "
            f"Found: {sorted(scenarios.tolist())}"
        )

    allowed_states = {
        "normal",
        "fault_active",
        "recovery",
    }

    states = set(dataframe["state"].astype(str).unique())
    unsupported_states = states.difference(allowed_states)

    if unsupported_states:
        raise DetectionDataError(
            "Scenario dataset contains unsupported state labels: "
            f"{sorted(unsupported_states)}"
        )

    return dataframe