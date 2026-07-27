import sys
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Project path setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "generated"
)

MANIFEST_PATH = (
    GENERATED_ROOT
    / "scenarios_manifest.csv"
)


REQUIRED_COLUMNS = {
    "timestamp_s",
    "asset_id",
    "asset_type",
    "parameter",
    "value",
    "scenario",
    "state",
}


def validate_scenario(
    row,
) -> list[str]:
    """
    Validate one generated scenario dataset.

    Returns a list of validation errors.
    """

    errors = []

    dataset_path = (
        PROJECT_ROOT
        / row["dataset_file"]
    )

    metadata_path = (
        PROJECT_ROOT
        / row["metadata_file"]
    )

    # --------------------------------------------------
    # Check files exist
    # --------------------------------------------------

    if not dataset_path.exists():
        errors.append(
            f"Dataset missing: {dataset_path}"
        )
        return errors

    if not metadata_path.exists():
        errors.append(
            f"Metadata missing: {metadata_path}"
        )

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    df = pd.read_csv(
        dataset_path
    )

    # --------------------------------------------------
    # Shape / structure checks
    # --------------------------------------------------

    missing_columns = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing_columns:
        errors.append(
            f"Missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(df) != int(row["rows"]):
        errors.append(
            f"Row count mismatch: "
            f"manifest={row['rows']} "
            f"actual={len(df)}"
        )

    if len(df) != 7825:
        errors.append(
            f"Unexpected dataset size: "
            f"{len(df)}"
        )

    # --------------------------------------------------
    # Missing / infinite values
    # --------------------------------------------------

    if df.isnull().any().any():
        errors.append(
            "Dataset contains missing values."
        )

    # --------------------------------------------------
    # Timestamp checks
    # --------------------------------------------------

    expected_timestamps = set(
        range(
            0,
            24 * 3600 + 1,
            3600,
        )
    )

    actual_timestamps = set(
        df["timestamp_s"].unique()
    )

    if actual_timestamps != expected_timestamps:
        errors.append(
            "Unexpected or missing timestamps."
        )

    # --------------------------------------------------
    # Scenario-name consistency
    # --------------------------------------------------

    scenario_values = (
        df["scenario"]
        .dropna()
        .unique()
    )

    if (
        len(scenario_values) != 1
        or scenario_values[0]
        != row["scenario"]
    ):
        errors.append(
            "Scenario column does not match manifest."
        )

    # --------------------------------------------------
    # Temporal-state checks
    # --------------------------------------------------

    start_s = (
        int(row["start_hour"])
        * 3600
    )

    end_s = (
        int(row["end_hour"])
        * 3600
    )

    state_by_time = (
        df[
            [
                "timestamp_s",
                "state",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "timestamp_s"
        )
    )

    for _, state_row in (
        state_by_time.iterrows()
    ):
        timestamp = int(
            state_row["timestamp_s"]
        )

        state = (
            state_row["state"]
        )

        if timestamp < start_s:
            expected_state = "normal"

        elif timestamp < end_s:
            expected_state = (
                "fault_active"
            )

        else:
            expected_state = "recovery"

        if state != expected_state:
            errors.append(
                f"Incorrect state at "
                f"{timestamp}s: "
                f"expected={expected_state}, "
                f"actual={state}"
            )

    return errors


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: "
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"\nGenerated scenarios in manifest: "
        f"{len(manifest)}"
    )

    # --------------------------------------------------
    # Manifest-level checks
    # --------------------------------------------------

    if len(manifest) != 35:
        print(
            f"WARNING: Expected 35 scenarios, "
            f"found {len(manifest)}."
        )

    duplicate_names = (
        manifest["scenario"]
        .duplicated()
        .sum()
    )

    if duplicate_names:
        print(
            f"ERROR: {duplicate_names} duplicate "
            "scenario names found."
        )
        return

    # --------------------------------------------------
    # Scenario validation
    # --------------------------------------------------

    failed = []

    for _, row in manifest.iterrows():

        errors = validate_scenario(
            row
        )

        if errors:
            failed.append(
                (
                    row["scenario"],
                    errors,
                )
            )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    if failed:

        print(
            f"\nFAILED scenarios: "
            f"{len(failed)}\n"
        )

        for scenario, errors in failed:

            print(
                f"[FAIL] {scenario}"
            )

            for error in errors:
                print(
                    f"    - {error}"
                )

    else:

        print(
            "\nAll generated scenario datasets "
            "passed validation."
        )

        print(
            "✓ Unique scenario names"
        )

        print(
            "✓ Expected row counts"
        )

        print(
            "✓ Required columns"
        )

        print(
            "✓ Complete timestamps"
        )

        print(
            "✓ No missing values"
        )

        print(
            "✓ Scenario labels"
        )

        print(
            "✓ Temporal state labels"
        )


if __name__ == "__main__":
    main()