"""Batch residual profiling across generated fault scenarios."""

from pathlib import Path

import pandas as pd

from twinrag.detection.aligner import align_baseline_and_scenario
from twinrag.detection.loader import (
    DetectionDataError,
    load_baseline_dataset,
    load_scenario_dataset,
)
from twinrag.detection.profiling import (
    rank_affected_assets,
    summarize_residuals_by_parameter,
)
from twinrag.detection.residuals import calculate_residuals


REQUIRED_MANIFEST_COLUMNS = {
    "scenario",
    "fault_type",
    "target_id",
    "severity",
    "start_hour",
    "end_hour",
    "dataset_file",
}


def _load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Load and validate the generated scenario manifest.

    Args:
        manifest_path: Path to ``scenarios_manifest.csv``.

    Returns:
        Validated manifest DataFrame.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        DetectionDataError: If the manifest is invalid.
    """
    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Scenario manifest does not exist: {path}"
        )

    if not path.is_file():
        raise DetectionDataError(
            f"Scenario manifest path is not a file: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise DetectionDataError(
            f"Scenario manifest must be a CSV file: {path}"
        )

    manifest = pd.read_csv(
        path,
        dtype={
            "scenario": str,
            "fault_type": str,
            "target_id": str,
            "dataset_file": str,
        },
    )

    if manifest.empty:
        raise DetectionDataError(
            "Scenario manifest contains no scenario records."
        )

    missing_columns = REQUIRED_MANIFEST_COLUMNS.difference(
        manifest.columns
    )

    if missing_columns:
        raise DetectionDataError(
            "Scenario manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    required_values = manifest[list(REQUIRED_MANIFEST_COLUMNS)]
    missing_values = required_values.isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        details = ", ".join(
            f"{column}={count}"
            for column, count in missing_values.items()
        )

        raise DetectionDataError(
            "Scenario manifest contains missing required values: "
            f"{details}"
        )

    if manifest["scenario"].duplicated().any():
        duplicate_names = (
            manifest.loc[
                manifest["scenario"].duplicated(keep=False),
                "scenario",
            ]
            .unique()
            .tolist()
        )

        raise DetectionDataError(
            "Scenario manifest contains duplicate scenario names: "
            f"{sorted(duplicate_names)}"
        )

    return manifest


def _resolve_scenario_path(
    dataset_file: str,
    processed_directory: str | Path,
) -> Path:
    """Resolve a manifest dataset reference to an existing CSV path.

    The manifest may contain either:

    - only the filename,
    - a relative path,
    - or an absolute path.

    Args:
        dataset_file: Value from the manifest's ``dataset_file`` column.
        processed_directory: Directory containing generated scenario CSVs.

    Returns:
        Resolved scenario dataset path.

    Raises:
        FileNotFoundError: If no valid path can be found.
    """
    manifest_path = Path(str(dataset_file))
    processed_path = Path(processed_directory)

    candidates = []

    if manifest_path.is_absolute():
        candidates.append(manifest_path)
    else:
        candidates.extend(
            [
                manifest_path,
                processed_path / manifest_path,
                processed_path / manifest_path.name,
            ]
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    candidate_text = ", ".join(str(path) for path in candidates)

    raise FileNotFoundError(
        "Could not locate scenario dataset. "
        f"Manifest value: {dataset_file}. "
        f"Checked: {candidate_text}"
    )


def _add_scenario_metadata(
    dataframe: pd.DataFrame,
    manifest_row: pd.Series,
) -> pd.DataFrame:
    """Add fault and scenario metadata to a profiling result."""
    result = dataframe.copy()

    metadata = {
        "scenario": str(manifest_row["scenario"]),
        "fault_type": str(manifest_row["fault_type"]),
        "target_id": str(manifest_row["target_id"]),
        "severity": float(manifest_row["severity"]),
        "start_hour": int(manifest_row["start_hour"]),
        "end_hour": int(manifest_row["end_hour"]),
    }

    for column, value in reversed(list(metadata.items())):
        result.insert(0, column, value)

    return result


def profile_all_scenarios(
    baseline_path: str | Path,
    manifest_path: str | Path,
    processed_directory: str | Path,
    top_n_assets: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile residual behaviour across all manifest scenarios.

    For each generated scenario, this function:

    1. Loads the scenario.
    2. Aligns it with the normal Digital Twin baseline.
    3. Calculates signed and absolute residuals.
    4. Summarizes residuals by state and parameter.
    5. Ranks the most strongly affected assets.

    Args:
        baseline_path: Path to the normal baseline CSV.
        manifest_path: Path to the generated scenario manifest.
        processed_directory: Directory containing generated scenario CSVs.
        top_n_assets: Number of asset-parameter rankings retained per
            scenario.

    Returns:
        A tuple containing:

        - combined parameter-level residual summary,
        - combined fault-active asset ranking.

    Raises:
        ValueError: If ``top_n_assets`` is not positive.
        DetectionDataError: If manifest and dataset scenario names differ.
    """
    if top_n_assets <= 0:
        raise ValueError(
            "top_n_assets must be greater than zero."
        )

    baseline = load_baseline_dataset(baseline_path)
    manifest = _load_manifest(manifest_path)

    parameter_summaries: list[pd.DataFrame] = []
    asset_rankings: list[pd.DataFrame] = []

    total_scenarios = len(manifest)

    for index, manifest_row in manifest.iterrows():
        scenario_name = str(manifest_row["scenario"])

        scenario_path = _resolve_scenario_path(
            dataset_file=str(manifest_row["dataset_file"]),
            processed_directory=processed_directory,
        )

        scenario_data = load_scenario_dataset(scenario_path)

        dataset_scenario_name = str(
            scenario_data["scenario"].iloc[0]
        )

        if dataset_scenario_name != scenario_name:
            raise DetectionDataError(
                "Manifest and dataset scenario names do not match. "
                f"Manifest: {scenario_name}; "
                f"dataset: {dataset_scenario_name}; "
                f"file: {scenario_path}"
            )

        aligned_data = align_baseline_and_scenario(
            baseline=baseline,
            scenario=scenario_data,
        )

        residual_data = calculate_residuals(aligned_data)

        parameter_summary = summarize_residuals_by_parameter(
            residual_data
        )

        parameter_summary = _add_scenario_metadata(
            dataframe=parameter_summary,
            manifest_row=manifest_row,
        )

        asset_ranking = rank_affected_assets(
            residual_data=residual_data,
            state="fault_active",
            top_n=top_n_assets,
        )

        asset_ranking.insert(
            0,
            "rank",
            range(1, len(asset_ranking) + 1),
        )

        asset_ranking = _add_scenario_metadata(
            dataframe=asset_ranking,
            manifest_row=manifest_row,
        )

        parameter_summaries.append(parameter_summary)
        asset_rankings.append(asset_ranking)

        print(
            f"[{index + 1:03d}/{total_scenarios:03d}] "
            f"Profiled {scenario_name}"
        )

    combined_parameter_summary = pd.concat(
        parameter_summaries,
        ignore_index=True,
    )

    combined_asset_ranking = pd.concat(
        asset_rankings,
        ignore_index=True,
    )

    return combined_parameter_summary, combined_asset_ranking