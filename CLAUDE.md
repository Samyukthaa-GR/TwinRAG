# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TwinRAG is a digital-twin platform for water distribution networks. The current codebase covers the **simulation layer** (Module 1 of a planned 6): it drives EPANET hydraulic simulations (via the `wntr` library) on `.inp` network models, injects faults to produce labelled fault-condition data, and formats results into standardized long-format datasets. The name implies an eventual RAG layer (anomaly detection → knowledge graph → topology-aware retrieval → LLM root-cause → dashboard) on top of the twin; those modules are not yet present.

Module 1 is complete: baseline simulation, **timed** leak / pump-failure / blockage fault injection, systematic scenario generation (a 35-scenario evaluation batch), temporal state labelling, and dataset validation. The remaining modules (2–6) are unbuilt — no `networkx`, LLM client, or Streamlit in `requirements.txt` yet.

## Commands

```bash
# Install deps (Python 3; use a venv — .venv/ and venv/ are gitignored)
pip install -r requirements.txt

# Inspect a network's node/link composition (sanity check the .inp loads)
python scripts/inspect_network.py

# Run the 24h baseline simulation -> writes data/processed/baseline.csv
python scripts/run_baseline_simulation.py

# Run baseline + every explicit fault in configs/simulation.yaml
# -> data/processed/<scenario>.csv + data/metadata/<scenario>.json per scenario
python scripts/run_fault_simulation.py
python scripts/run_fault_simulation.py --config configs/simulation.yaml --skip-baseline

# --- Generated evaluation batch (scenario_generation block in the config) ---

# Print the generated scenario matrix without simulating anything (fast, no wntr run)
python scripts/inspect_generated_scenarios.py

# Check every generated target_id actually exists in the network, with the right asset kind
python scripts/validate_generated_scenarios.py

# Simulate the whole generated batch -> data/generated/{processed,metadata}/
# plus data/generated/scenarios_manifest.csv (one row per scenario)
python scripts/run_generated_scenarios.py

# Post-hoc validation of that batch against the manifest (schema, row counts,
# timestamps, nulls, scenario labels, temporal state labels)
python scripts/validate_generated_dataset.py

# Tests (pytest.ini sets pythonpath=src and testpaths=tests — no sys.path juggling needed)
pytest
pytest tests/simulation/test_scenario_generator.py::test_generated_scenario_count   # single test
```

On Windows use the `py` launcher and the venv interpreter explicitly: `py -m venv .venv`, then `.\.venv\Scripts\python.exe ...`. The bare `py`/`python` on PATH does *not* have the dependencies installed. The full suite (9 tests) and real EPANET simulations both run in this environment.

Note: the machine's Application Control policy intermittently blocks pandas' compiled DLLs on first touch — `ImportError: DLL load failed while importing <ext>: An Application Control policy has blocked this file`. It has cleared as of the latest check (pandas 3.0.5 imports, `pytest` is 9/9 green, `run_baseline_simulation.py` completes). If it reappears, retrying the same command usually succeeds; it is an OS-level policy quirk, not a code or dependency problem.

There is no `pyproject.toml`/`setup.py`. The package is **not installed** — scripts add `src/` to `sys.path` manually (see the path-setup block at the top of each script) before importing `twinrag.*`. New entry-point scripts must replicate that block, and the `twinrag` imports must come *after* it. Tests don't need it: `pytest.ini` sets `pythonpath = src`.

## Architecture

Pipeline: **load → configure → simulate → format → persist**, orchestrated by scripts in `scripts/`. The `src/twinrag/simulation/` package holds one class per stage, each independently testable:

- `WaterNetworkLoader` (`network_loader.py`) — validates and loads an EPANET `.inp` into a `wntr` `WaterNetworkModel`. Also exposes `get_summary()`/`print_summary()` for network composition.
- `HydraulicSimulator` (`simulator.py`) — wraps `wntr.sim.EpanetSimulator`. `configure_simulation()` sets duration/timesteps (defaults: 24h, 3600s steps), `run()` executes, and `get_pressure()`/`get_demand()`/`get_flowrate()` return copies of the result time-series. Result accessors raise if `run()` hasn't been called.
- `SimulationResultFormatter` (`result_formatter.py`) — melts WNTR's wide dataframes (rows = timestamps, cols = asset IDs) into a long format and `combine()`s pressure/demand/flowrate into one dataset. This long schema is the contract downstream code depends on.
- `FaultScenarioRunner` (`scenario.py`) — orchestrates one scenario end-to-end: load → (inject fault) → simulate → format → **label temporal state** → return `(dataset, metadata)`. **Loads a fresh network per scenario** so in-place fault mutations never leak between runs.
- `config.py` — YAML-backed dataclasses (`load_config`): `ExperimentConfig` (network + `SimulationConfig` + explicit `faults` + `scenario_generation`), plus `FaultScenarioConfig`, `FaultGenerationConfig`, `ScenarioGenerationConfig`. Validation lives here: severity in `(0.0, 1.0]`, `start_hour >= 0`, `end_hour > start_hour`, and — for generated blocks — `start_hour + duration_hours <= simulation.duration_hours`.
- `scenario_generator.py` — `generate_fault_scenarios(generation_config)` expands each fault block into the Cartesian product `target_ids × severities × start_hours`, deriving `end_hour = start_hour + duration_hours`. Returns `[]` when `scenario_generation.enabled` is false. Deterministic (no RNG), so the batch is reproducible. The current `configs/simulation.yaml` yields **35 scenarios** (27 leak, 2 pump_failure, 6 blockage) — `tests/simulation/test_scenario_generator.py` asserts those counts, so changing the YAML matrix means updating those tests.

### Fault injection (`simulation/faults/`)

`FaultInjector` (`base.py`) is an abstract base: `apply(network)` runs `_validate_target` then `_inject`, mutating the `WaterNetworkModel` **in place**. Constructor args are `target_id`, `severity` in `(0.0, 1.0]`, `start_hour`, `end_hour` (`None` = fault runs to the end of the simulation). Subclasses set a `fault_type`, and `scenario_name()` derives a stable, collision-free label that encodes the time window — `leak_101_sev60_t08_18`, or `..._t08_end` when `end_hour is None` — used for both the `scenario` column and output filenames. `describe()` returns ground-truth metadata (mechanism, severity, time window, affected node/link IDs) for downstream anomaly-detection labelling.

Faults are **timed**: the injected condition switches on at `start_hour` and off at `end_hour` *within the hydraulics*, via EPANET-compatible patterns and rules — not by mutating a static property for the whole run. All three are **EpanetSimulator-compatible** (important: WNTR's `Junction.add_leak` only works under `WNTRSimulator`, which this project does *not* use):

- `LeakFault` (`leak.py`) — adds a **second demand** to the junction (`add_demand(..., category="Leak")`) driven by a 0/1 pattern that is `1.0` only inside the fault window. Leak magnitude is `severity × max_leak_demand` (default `max_leak_demand=0.05`). The junction's original demand is left untouched.
- `PumpFailureFault` (`pump_failure.py`) — at `severity >= 1.0` uses WNTR's `pump.add_outage(network, start_time, end_time, priority=6, add_after_outage_rule=True)` for a genuine timed outage. **Partial severity is still untimed** — it just scales `base_speed` by `(1 − severity)` for the entire run; the start/end hours are recorded in metadata but not honoured hydraulically.
- `BlockageFault` (`valve_blockage.py`) — targets any link (Net3 has **no valves**, so a pipe). Adds two WNTR `Rule`s on `SimTimeCondition`: close the link at `start_hour` (priority 6), reopen at `end_hour` (priority 7). **Requires `end_hour`** — raises `ValueError` without it. Note `severity` does not affect the injection at all; the link is fully closed regardless, so severity only varies the scenario label.

Build faults via `create_fault(fault_type, target_id, severity, start_hour, end_hour, **params)` from `faults/__init__.py` (backed by `FAULT_REGISTRY`); extra params forward to the injector (e.g. `max_leak_demand`).

### The dataset schema

`combine()` produces the long-format core, one row per (timestamp, asset, parameter); `FaultScenarioRunner` then appends `state`:

```
timestamp_s, asset_id, asset_type, parameter, value, scenario, state
```

- `asset_type` is `node` (pressure, demand) or `link` (flowrate).
- `scenario` is set from `SimulationResultFormatter(scenario_name=...)` — `"normal"` for baseline, `fault.scenario_name()` otherwise. All datasets share this schema and can be concatenated.
- `state` is the temporal label used as anomaly-detection ground truth: `normal` before `start_hour`, `fault_active` for `start_hour <= t < end_hour`, `recovery` for `t >= end_hour` (recovery is distinct from normal because tank levels and hydraulic state have drifted during the fault). With `end_hour is None` the fault stays `fault_active` to the end. Baseline runs are labelled `normal` throughout.
- **Gotcha:** `state` is added by `FaultScenarioRunner`, not the formatter. `scripts/run_baseline_simulation.py` calls the loader/simulator/formatter directly and therefore writes a `data/processed/baseline.csv` *without* the `state` column, unlike the baseline written by `run_fault_simulation.py`. Anything consuming the schema uniformly should prefer the runner.
- For Net3 at the default 24h/3600s settings each scenario dataset is **7825 rows** = 25 timestamps × (97 nodes × 2 params + 119 links); `validate_generated_dataset.py` hard-codes that number and the expected 35-scenario count.

### Configuration (`configs/simulation.yaml`)

Two independent ways to define scenarios, both read from the same file:

- `faults:` — an explicit hand-written list, consumed by `run_fault_simulation.py`.
- `scenario_generation:` — the combinatorial matrix (`enabled`, then a `leak`/`pump_failure`/`blockage` block each with `target_ids`, `severities`, `start_hours`, `duration_hours`), consumed by the `*_generated_*` scripts.

The two do not interact — `run_fault_simulation.py` ignores `scenario_generation`, and the generated-batch scripts ignore `faults`.

### Data layout

- `data/networks/*.inp` — input EPANET models (committed; e.g. `Net3.inp`).
- `data/processed/` — datasets from the baseline/explicit-fault runs (gitignored except `.gitkeep`).
- `data/metadata/` — per-scenario ground-truth fault labels as `<scenario>.json` (gitignored), written alongside each fault dataset by `run_fault_simulation.py`.
- `data/generated/processed/`, `data/generated/metadata/` — the generated evaluation batch, same file naming, written by `run_generated_scenarios.py` (gitignored).
- `data/generated/scenarios_manifest.csv` — index of the batch, one row per scenario: `scenario_id, scenario, fault_type, target_id, severity, start_hour, end_hour, dataset_file, metadata_file, rows`. `validate_generated_dataset.py` drives entirely off this file, so it is the entry point for consuming the batch.

Generated simulation artifacts (`*.bin`, `*.hyd`, `*.rpt`, and the contents of `data/processed`, `data/metadata`, `data/generated`) are gitignored — do not commit them.

### Tests

`tests/simulation/` mirrors the package layout. Existing coverage: `test_network_loader.py` (Net3 loads; exact component counts — 92 junctions / 2 reservoirs / 3 tanks / 117 pipes / 2 pumps; missing file raises), `test_scenario_generator.py` (generated counts, valid time windows, unique scenario names), `test_scenario_labels.py` (the `normal`/`fault_active`/`recovery` boundaries, using a `SimpleNamespace` stub fault and a hand-built dataframe rather than a real simulation). Tests call the runner's private `_add_*_state_labels` helpers directly to stay fast — keep those names stable or update the tests.
