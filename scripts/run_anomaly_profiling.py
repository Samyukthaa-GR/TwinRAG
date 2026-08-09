"""Generate residual profiling reports for all fault scenarios."""

import sys
from pathlib import Path


# --------------------------------------------------
# Project path setup
# The package is not installed, so src/ has to be on sys.path before
# any twinrag import.
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from twinrag.detection.batch import profile_all_scenarios


BASELINE_PATH = Path("data/processed/baseline.csv")
MANIFEST_PATH = Path("data/generated/scenarios_manifest.csv")
PROCESSED_DIRECTORY = Path("data/generated/processed")
OUTPUT_DIRECTORY = Path("data/detection")

PARAMETER_OUTPUT_PATH = (
    OUTPUT_DIRECTORY / "parameter_residual_summary.csv"
)

ASSET_OUTPUT_PATH = (
    OUTPUT_DIRECTORY / "affected_asset_ranking.csv"
)


def main() -> None:
    """Run batch residual profiling and save the resulting reports."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("TwinRAG Module 2 — Batch Residual Profiling")
    print(f"Baseline: {BASELINE_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Scenario directory: {PROCESSED_DIRECTORY}")
    print()

    parameter_summary, asset_ranking = profile_all_scenarios(
        baseline_path=BASELINE_PATH,
        manifest_path=MANIFEST_PATH,
        processed_directory=PROCESSED_DIRECTORY,
        top_n_assets=20,
    )

    parameter_summary.to_csv(
        PARAMETER_OUTPUT_PATH,
        index=False,
    )

    asset_ranking.to_csv(
        ASSET_OUTPUT_PATH,
        index=False,
    )

    print()
    print("Batch residual profiling completed.")
    print(
        "Parameter summary rows: "
        f"{len(parameter_summary)}"
    )
    print(
        "Asset-ranking rows: "
        f"{len(asset_ranking)}"
    )
    print(
        "Parameter report: "
        f"{PARAMETER_OUTPUT_PATH}"
    )
    print(
        "Asset-ranking report: "
        f"{ASSET_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()