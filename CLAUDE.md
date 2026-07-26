# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TwinRAG is a digital-twin platform for water distribution networks. The current codebase covers the **simulation layer** (Module 1 of a planned 6): it drives EPANET hydraulic simulations (via the `wntr` library) on `.inp` network models, injects faults to produce labelled fault-condition data, and formats results into standardized long-format datasets. The name implies an eventual RAG layer (anomaly detection → knowledge graph → topology-aware retrieval → LLM root-cause → dashboard) on top of the twin; those modules are not yet present.

Module 1 is complete: baseline simulation plus leak / pump-failure / blockage fault injection. The remaining modules (2–6) are unbuilt — no `networkx`, LLM client, or Streamlit in `requirements.txt` yet.

## Commands

```bash
# Install deps (Python 3; use a venv — .venv/ and venv/ are gitignored)
pip install -r requirements.txt

# Inspect a network's node/link composition (sanity check the .inp loads)
python scripts/inspect_network.py

# Run the 24h baseline simulation -> writes data/processed/baseline.csv
python scripts/run_baseline_simulation.py

# Run baseline + every fault in configs/simulation.yaml
# -> data/processed/<scenario>.csv + data/metadata/<scenario>.json per scenario
python scripts/run_fault_simulation.py
python scripts/run_fault_simulation.py --config configs/simulation.yaml --skip-baseline

# Tests (pytest is a dependency but no tests exist yet; place them under tests/)
pytest
pytest tests/test_simulator.py::TestName::test_case   # single test
```

On Windows use the `py` launcher (`py -m venv .venv`; `.\.venv\Scripts\python.exe ...`). Note: an OS Application Control policy on the current machine blocks pandas' compiled DLL, so wntr/pandas cannot be imported/run here — the fault-injection and config logic (which don't import those) can still be unit-tested with a stub network, but full simulations must run on an unrestricted environment.

There is no `pyproject.toml`/`setup.py`. The package is **not installed** — scripts add `src/` to `sys.path` manually (see the path-setup block at the top of each script) before importing `twinrag.*`. New entry-point scripts must replicate that block, and the `twinrag` imports must come *after* it.

## Architecture

Pipeline: **load → configure → simulate → format → persist**, orchestrated by scripts in `scripts/`. The `src/twinrag/simulation/` package holds one class per stage, each independently testable:

- `WaterNetworkLoader` (`network_loader.py`) — validates and loads an EPANET `.inp` into a `wntr` `WaterNetworkModel`. Also exposes `get_summary()`/`print_summary()` for network composition.
- `HydraulicSimulator` (`simulator.py`) — wraps `wntr.sim.EpanetSimulator`. `configure_simulation()` sets duration/timesteps (defaults: 24h, 3600s steps), `run()` executes, and `get_pressure()`/`get_demand()`/`get_flowrate()` return copies of the result time-series. Result accessors raise if `run()` hasn't been called.
- `SimulationResultFormatter` (`result_formatter.py`) — melts WNTR's wide dataframes (rows = timestamps, cols = asset IDs) into a long format and `combine()`s pressure/demand/flowrate into one dataset. This long schema is the contract downstream code depends on.
- `FaultScenarioRunner` (`scenario.py`) — orchestrates one scenario end-to-end: load → (inject fault) → simulate → format → return `(dataset, metadata)`. **Loads a fresh network per scenario** so in-place fault mutations never leak between runs. `config.py` holds the YAML-backed `ExperimentConfig`/`SimulationConfig`/`FaultScenarioConfig` dataclasses (`load_config`).

### Fault injection (`simulation/faults/`)

`FaultInjector` (`base.py`) is an abstract base: `apply(network)` runs `_validate_target` then `_inject`, mutating the `WaterNetworkModel` **in place**. `severity` is `(0.0, 1.0]` where `1.0` is the most severe form. Subclasses set a `fault_type`, and `scenario_name()` derives a stable label like `leak_101_sev60` used for both the `scenario` column and output filenames. `describe()` returns ground-truth metadata (mechanism + affected node/link IDs) for downstream anomaly-detection labelling.

Three concrete faults, all **EpanetSimulator-compatible** (important: WNTR's `Junction.add_leak` only works under `WNTRSimulator`, which this project does *not* use):
- `LeakFault` (`leak.py`) — sets a junction's `emitter_coefficient` (`= severity × max_emitter_coefficient`); EPANET emitter models pressure-driven leak outflow.
- `PumpFailureFault` (`pump_failure.py`) — full severity closes the pump (`initial_status="Closed"`); partial reduces `base_speed` to `(1−severity)`.
- `BlockageFault` (`valve_blockage.py`) — targets any link (Net3 has **no valves**, so a pipe): partial adds `severity × max_minor_loss` to `minor_loss`, full closes the link.

Build faults via `create_fault(fault_type, target_id, severity, **params)` from `faults/__init__.py` (backed by `FAULT_REGISTRY`); extra params forward to the injector (e.g. `max_emitter_coefficient`, `max_minor_loss`).

### The dataset schema

`combine()` produces the canonical output, one row per (timestamp, asset, parameter):

```
timestamp_s, asset_id, asset_type, parameter, value, scenario
```

- `asset_type` is `node` (pressure, demand) or `link` (flowrate).
- `scenario` is set from `SimulationResultFormatter(scenario_name=...)` — `"normal"` for baseline. Fault scenarios should reuse this same formatter with a distinct scenario name so all datasets share one schema and can be concatenated.

### Data layout

- `data/networks/*.inp` — input EPANET models (committed; e.g. `Net3.inp`).
- `data/processed/` — generated datasets (gitignored except `.gitkeep`).
- `data/metadata/` — per-scenario ground-truth fault labels as `<scenario>.json` (gitignored), written alongside each fault dataset by `run_fault_simulation.py`.

Generated simulation artifacts (`*.bin`, `*.hyd`, `*.rpt`, and `data/processed`/`data/metadata` contents) are gitignored — do not commit them.
