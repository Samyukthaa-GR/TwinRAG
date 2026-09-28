# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TwinRAG is a digital-twin platform for water distribution networks. The end goal: an anomaly fires somewhere in the network, a subgraph around that point is extracted from the knowledge graph, and that local physical context is handed to an LLM which reasons out the root cause with no manual pipe-tracing. The simulator exists to manufacture ground truth for that pipeline.

Two networks run through the same pipeline:

- **The building twin** (`data/networks/Building_G6.inp`) — the project's demo system and current direction. A generated, constructible G+6 residential block: 7 floors x 2 flats, room-level plumbing (70 wet-room plumbing points), sump -> transfer pump -> roof tank -> gravity down-take risers, with a 3D viewer. See *Building digital twin* below.
- **Net3** (`data/networks/Net3.inp`) — EPANET's example town network. The Phase 3 evaluation numbers below were measured on it, and `configs/simulation.yaml` still targets it.

Six planned phases. **Phases 1–4 are built** (Phase 4 for the building only):

1. **Simulation** — EPANET/`wntr` runs on `.inp` models, timed fault injection, long-format datasets. Complete.
2. **Knowledge graph** — `networkx` topology, per-timestep hydraulic snapshots with flow direction and neighbour lookup, plus a schema/validation layer. Complete.
3. **Anomaly detection** — **two parallel implementations, both kept.** See *Anomaly detection* below; this is the single most important thing to understand before touching the package.
4. **Knowledge graph + topology-aware retrieval** — typed building knowledge graph (`twinrag.graph.knowledge`) and physics-guided subgraph retrieval around an incident, serialized as a leak-checked LLM evidence packet (`twinrag.retrieval`). Built for the building; Net3 is not supported (it is looped, see below).
5. **Root-cause diagnosis** — **Decided (Sept 2026): rule-based diagnosis messages are the primary output**, generated deterministically from the evidence packet's topology facts (not built yet — next task). The LLM path (`twinrag.reasoning`, built) stays as an *optional* comparison, run only via `scripts/diagnose_incidents.py`; nothing else calls it, so it is effectively off. Reason: on a 6-incident sample gpt-oss-120b (Groq free tier) got leaks to the right room and the afternoon riser blockage exactly, but called both pump failures a roof-tank leak even with the time-ordered evidence, and is rate-limited. Knowledge graph stays in NetworkX; a Neo4j/Cypher export is planned for later.
6. **Operator dashboard** — a static prototype viewer exists (see *Viewer* below); the live backend, anomaly overlay and diagnosis panel do not.

> **Two Phase 3 pipelines coexist in `src/twinrag/detection/`** — a dataclass/object pipeline and a DataFrame/CSV pipeline, built independently by two contributors and merged deliberately rather than consolidated. They share no types. Pick one per consumer; do not interleave them. **Phase 4 consumes pipeline A's `Incident`** (ranked candidates keyed by asset, tested, sensor-noise aware); pipeline B's event rows join node and link IDs into one string and are not used downstream.

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

# --- Phase 3, pipeline A (dataclass): detect -> evaluate ---

# Run a detector over all 35 scenarios + the baseline negative control
# -> data/generated/detection/<name>/{summary.csv,incidents.json,run.json,events/}
python scripts/detect_anomalies.py
python scripts/detect_anomalies.py --detector statistical
python scripts/detect_anomalies.py --threshold 6 --min-assets 5
python scripts/detect_anomalies.py --no-noise    --name residual_noisefree
python scripts/detect_anomalies.py --noise-scale 3 --name residual_noise3x

# Score a run against ground truth (recall, precision, latency, hit@k, hops)
# -> adds evaluation.csv + evaluation.json to that run's directory
python scripts/evaluate_detection.py --name residual

# --- Phase 3, pipeline B (DataFrame): run these four IN ORDER ---
# They are cwd-sensitive: run from the project root, since they resolve
# paths like Path("data/processed/baseline.csv") relative to the cwd.

python scripts/run_anomaly_detection.py    # -> data/detection/scenario_detection_summary.csv
python scripts/run_anomaly_profiling.py    # -> parameter_residual_summary + affected_asset_ranking
python scripts/run_event_aggregation.py    # -> anomaly_events.csv + event_evidence.csv
python scripts/run_detection_evaluation.py # -> data/evaluation/*  (reads the CSVs above)

# --- Viewer (static prototype of the Phase 6 dashboard) ---

python scripts/export_graph_view.py   # topology + 4 scenarios -> graph_view.json
python scripts/build_viewer.py        # -> data/generated/twin_viewer.html
python scripts/server_viewer.py       # optional; serves it over HTTP

# --- Building digital twin ---

python scripts/build_building_network.py   # -> data/networks/Building_G6.{inp,layout.json}
python scripts/run_generated_scenarios.py --config configs/building.yaml \
    --output-root data/building --with-baseline   # 38 scenarios + baseline -> data/building/
python scripts/build_building_viewer.py    # -> data/building/building_twin.html (3D, offline)

# --- Phase 4 on the building: detect -> knowledge graph -> retrieval ---
# (needs the building batch above). Evidence packets + retrieval_summary.csv
# -> data/building/evidence/ ; prints retrieval hit rates per fault type.
python scripts/build_evidence_packets.py
python scripts/build_evidence_packets.py --hops 3 --threshold 5

# --- Phase 5: LLM diagnosis of those packets (needs LLM settings in .env) ---
# -> data/building/diagnosis/<name>/{<scenario>.json,summary.csv,run.json}
python scripts/diagnose_incidents.py --per-type 2 --delay 45        # cheap sample
python scripts/diagnose_incidents.py --delay 45 --name gptoss        # all incidents
python scripts/diagnose_incidents.py --delay 45 --no-graph --name gptoss-nograph

# Tests (pytest.ini sets pythonpath=src and testpaths=tests — no sys.path juggling needed)
pytest
pytest tests/detection                                                             # one package
pytest tests/simulation/test_scenario_generator.py::test_generated_scenario_count   # single test
```

**Python 3.13, not 3.14.** `wntr` 1.5 ships no wheel for 3.14 and its source build needs MSVC, so on a machine whose only Python is 3.14 the install fails. The `.venv` here was built with `uv` (which downloads a self-contained 3.13): `python -m pip install uv`, `python -m uv venv .venv --python 3.13`, `python -m uv pip install --python .venv/Scripts/python.exe -r requirements.txt`. Then always call `.\.venv\Scripts\python.exe ...`; the bare `py`/`python` on PATH has no dependencies. The full suite runs here; the simulation-dependent Net3 graph tests skip if `data/processed/baseline.csv` has not been generated (`run_baseline_simulation.py`).

**Simulations dirty the working tree.** WNTR's `EpanetSimulator` writes its scratch `temp.inp` into the current directory, and `temp.inp` plus three `__pycache__/*.pyc` files are (unfortunately) tracked. After running any simulation from the project root, `git checkout -- temp.inp src/twinrag/__pycache__ src/twinrag/simulation/__pycache__` before committing. The building tests `chdir` into a temp directory to avoid this.

On Windows the console is cp1252, so scripts that print a check mark (`validate_generated_dataset.py`) crash with `UnicodeEncodeError` — prefix `PYTHONIOENCODING=utf-8`.

Note: the machine's Application Control policy intermittently blocks pandas' compiled DLLs on first touch — `ImportError: DLL load failed while importing <ext>: An Application Control policy has blocked this file`. It has cleared as of the latest check (pandas 3.0.5 imports, the suite is green, `run_baseline_simulation.py` completes). If it reappears, retrying the same command usually succeeds; it is an OS-level policy quirk, not a code or dependency problem.

There is no `pyproject.toml`/`setup.py`. The package is **not installed** — scripts add `src/` to `sys.path` manually (see the path-setup block at the top of each script) before importing `twinrag.*`. New entry-point scripts must replicate that block, and the `twinrag` imports must come *after* it. Tests don't need it: `pytest.ini` sets `pythonpath = src`.

## Architecture

Pipeline: **load → configure → simulate → format → persist**, orchestrated by scripts in `scripts/`. The `src/twinrag/simulation/` package holds one class per stage, each independently testable:

- `WaterNetworkLoader` (`network_loader.py`) — validates and loads an EPANET `.inp` into a `wntr` `WaterNetworkModel`. Also exposes `get_summary()`/`print_summary()` for network composition.
- `HydraulicSimulator` (`simulator.py`) — wraps `wntr.sim.EpanetSimulator`. `configure_simulation()` sets duration/timesteps (defaults: 24h, 3600s steps), `run()` executes, and `get_pressure()`/`get_demand()`/`get_flowrate()` return copies of the result time-series. Result accessors raise if `run()` hasn't been called.
- `SimulationResultFormatter` (`result_formatter.py`) — melts WNTR's wide dataframes (rows = timestamps, cols = asset IDs) into a long format and `combine()`s pressure/demand/flowrate into one dataset. This long schema is the contract downstream code depends on.
- `FaultScenarioRunner` (`scenario.py`) — orchestrates one scenario end-to-end: load → (inject fault) → simulate → format → **label temporal state** → return `(dataset, metadata)`. **Loads a fresh network per scenario** so in-place fault mutations never leak between runs.
- `config.py` — YAML-backed dataclasses (`load_config`): `ExperimentConfig` (network + `SimulationConfig` + explicit `faults` + `scenario_generation`), plus `FaultScenarioConfig`, `FaultGenerationConfig`, `ScenarioGenerationConfig`. Validation lives here: severity in `(0.0, 1.0]`, `start_hour >= 0`, `end_hour > start_hour`, and — for generated blocks — `start_hour + duration_hours <= simulation.duration_hours`. Both explicit faults and generation blocks accept `params:`, forwarded to the injector. `SimulationConfig.report_average` (default `False`) simulates at the hydraulic step and reports each period's **mean** over `(t - period, t]` (what a logging meter records) instead of an instantaneous snapshot — same number of rows. The building sets it (with 300 s hydraulic steps) because hourly snapshots missed the ~40-minute pump runs entirely, so a stopped pump was invisible. `SimulationConfig.pressure_floor_m` (default `None`) clips reported pressure (before averaging): EPANET returns meaningless negative heads (-9 to -12 m) for sections cut off from every source, which a tree-shaped building produces whenever a riser closes; the building config sets it to `0.0` (a drained pipe reads atmospheric).
- `scenario_generator.py` — `generate_fault_scenarios(generation_config)` expands each fault block into the Cartesian product `target_ids × severities × start_hours`, deriving `end_hour = start_hour + duration_hours`. Returns `[]` when `scenario_generation.enabled` is false. Deterministic (no RNG), so the batch is reproducible. The current `configs/simulation.yaml` yields **35 scenarios** (27 leak, 2 pump_failure, 6 blockage) — `tests/simulation/test_scenario_generator.py` asserts those counts, so changing the YAML matrix means updating those tests.

### Fault injection (`simulation/faults/`)

`FaultInjector` (`base.py`) is an abstract base: `apply(network)` runs `_validate_target` then `_inject`, mutating the `WaterNetworkModel` **in place**. Constructor args are `target_id`, `severity` in `(0.0, 1.0]`, `start_hour`, `end_hour` (`None` = fault runs to the end of the simulation). Subclasses set a `fault_type`, and `scenario_name()` derives a stable, collision-free label that encodes the time window — `leak_101_sev60_t08_18`, or `..._t08_end` when `end_hour is None` — used for both the `scenario` column and output filenames. `describe()` returns ground-truth metadata (mechanism, severity, time window, affected node/link IDs) for downstream anomaly-detection labelling.

Faults are **timed**: the injected condition switches on at `start_hour` and off at `end_hour` *within the hydraulics*, via EPANET-compatible patterns and rules — not by mutating a static property for the whole run. All three are **EpanetSimulator-compatible** (important: WNTR's `Junction.add_leak` only works under `WNTRSimulator`, which this project does *not* use):

- `LeakFault` (`leak.py`) — adds a **second demand** to the junction (`add_demand(..., category="Leak")`) driven by a 0/1 pattern that is `1.0` only inside the fault window. Leak magnitude is `severity × max_leak_demand` (default `max_leak_demand=0.05`). The junction's original demand is left untouched.
- `PumpFailureFault` (`pump_failure.py`) — at `severity >= 1.0` uses WNTR's `pump.add_outage(network, start_time, end_time, priority=6, add_after_outage_rule=restore_after_outage)` for a genuine timed outage. `restore_after_outage` defaults to `True` (Net3's behaviour), but WNTR's after-outage rule holds the pump **open for the rest of the run** at priority 6, outranking any level control — so a tank-switched pump (the building's `PUMP1`) must pass `restore_after_outage: false`. **Partial severity is still untimed** — it just scales `base_speed` by `(1 − severity)` for the entire run; the start/end hours are recorded in metadata but not honoured hydraulically.
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

### The graph layer (`src/twinrag/graph/`) — Phase 2

- `NetworkGraphBuilder` (`builder.py`) — turns a `WaterNetworkModel` into an undirected, keyed `nx.MultiGraph` (edge key = link ID, so parallel links are safe). Junctions/tanks/reservoirs become nodes (reservoir `base_head` is normalised onto the same `elevation` field as junction elevation); pipes/pumps/valves become edges. Every node and edge carries three attribute sections from `schema.py` — `static` (from EPANET), `runtime` (filled by a snapshot) and `anomaly` (empty; reserved for Phase 4) — plus flat compatibility fields. `summary(graph)` gives composition + connectivity.
- The graph is deliberately **undirected**: 56 of Net3's 119 links reverse over a normal 24h day as tanks fill and drain, so direction is a property of a moment, not of the network.
- `GraphSnapshot` (`snapshot.py`) — lays one timestep over the topology: pressure/demand onto nodes, flowrate onto links, and `flow_from`/`flow_to` derived from the **sign** of each flowrate. `directed()` returns the flow-oriented `MultiDiGraph` (links with *missing* flow are omitted; zero-flow links keep their declared orientation); `neighbors(asset_id, hops)` is the BFS retrieval primitive Phase 4 builds on; `upstream()`/`downstream()` give causal ancestry and blast radius at that instant. **Leakage hazard for Phase 4/5:** the snapshot copies `runtime_scenario` and `runtime_state` into `graph.graph`, and scenario names encode the answer (`leak_F3A-KIT_sev100_...`) — strip both before anything reaches an LLM.

### Anomaly detection (`src/twinrag/detection/`) — Phase 3

**Read this before editing the package.** Two independent implementations live here, merged from two contributors and kept both. They share no types and have separate entry-point scripts. `__init__.py` re-exports both, so `from twinrag.detection import X` works regardless of which file `X` lives in.

| | pipeline A — dataclass | pipeline B — DataFrame |
|---|---|---|
| modules | `sensors`, `base`, `residual`, `statistical`, `incidents` | `loader`, `aligner`, `residuals`, `detector`, `events`, `profiling`, `batch`, `config` |
| interface | `detect(dataset) -> AnomalyReport` | functions returning `pd.DataFrame` |
| thresholds | σ of sensor noise (`--threshold 4`) | absolute per-parameter values in `config.py` |
| output unit | `Incident` (window + ranked candidates) | event rows in `anomaly_events.csv` |
| scripts | `detect_anomalies.py`, `evaluate_detection.py` | the four `run_*.py`, in order |
| tests | `tests/detection/` (35 tests) | none yet |

**Naming trap:** `events.py` belongs to pipeline B (event *aggregation*, 567 lines). Pipeline A's `AnomalyEvent`/`Incident`/`AnomalyReport` dataclasses are in **`incidents.py`**. Both sides originally shipped an `events.py`; that was the merge conflict, resolved by moving pipeline A's to `incidents.py` and leaving pipeline B's untouched.

Pipeline B's distinctive pieces, worth knowing even if you work in A:

- `config.py` — `ParameterThreshold(anomaly, warning, critical)` severity tiers; pressure `0.50/1.50/5.00` m, flowrate `0.005/0.020/0.100`. Pipeline A has no tiers, only a continuous σ score — Phase 6 will want bands like these.
- `classify_event_phase(ground_truth_states)` — labels an event `primary_fault_event` / `recovery_transient` / `pre_fault_false_positive` / `mixed_or_unknown`. A cleaner formalisation of the recovery-vs-false-alarm problem than pipeline A's "report recovery separately". **Uses ground truth, so it is offline-evaluation only** — its docstring says so explicitly; never call it from a deployed detector or from Phase 4/5.
- `loader.py` — defensive dataset validation (`DetectionDataError`, `REQUIRED_COLUMNS`, `SUPPORTED_PARAMETERS`).
- Pipeline B reports **35/35 scenarios detected, 0 pre-fault anomalies, max detection delay 3600 s, 22 recovery transients**. Its outputs are deterministic — regenerating them reproduces the committed CSVs byte-for-byte.
- Caveat: pipeline B uses **absolute** thresholds against noise-free residuals. Because EPANET is deterministic, pre-fault residuals are exactly `0.000000`, so its zero-false-positive result is structural rather than earned. Pipeline A's `SensorModel` exists specifically to remove that free pass — see below.

Two fixes were applied to pipeline B during the merge, both pre-existing on its branch:
1. `detection/__init__.py` imported `.builder`/`.snapshot`/`.validation`, which live in `twinrag.graph`, not here — so `import twinrag.detection` raised `ImportError`. Those four names are now re-exported from their real home for compatibility; `twinrag.graph` is canonical.
2. `run_anomaly_detection.py`, `run_anomaly_profiling.py` and `run_event_aggregation.py` were missing the `sys.path` bootstrap block, so all three died with `ModuleNotFoundError: No module named 'twinrag'`. `run_detection_evaluation.py` survived only because it imports nothing from `twinrag`.

The rest of this section documents **pipeline A**.

- `SensorModel` (`sensors.py`) — **required for the evaluation to mean anything.** EPANET is deterministic, so a fault run and the baseline agree to the last bit until the fault fires; pre-fault residuals are *exactly* `0.000000`. Without noise, any threshold above zero scores 100% precision while measuring nothing. `observe()` adds seeded Gaussian noise (`absolute + relative × |value|` per parameter, defaults in `DEFAULT_NOISE`: pressure 0.30 m, flowrate 0.001 + 1%) and optionally restricts to instrumented assets. It keeps `truth` alongside the noisy `value` and publishes `noise_std`, which is what detectors divide by. Noise attaches to a reading's identity via a canonically sorted frame, so row order cannot change the data.
- `ResidualDetector` (`residual.py`) — **the primary method.** `score = |observed − twin_expected| / noise_std`, so the threshold is in standard deviations and one setting works across metres and m³/s. `expected` comes from the baseline dataset (its `truth` column when present, so a baseline that has been through a sensor model still contributes noise-free values). Optional `min_residual` floors the raw deviation, stopping a very precise sensor from turning a hydraulically trivial wobble into an alarm.
- `StatisticalDetector` (`statistical.py`) — the baseline-free ablation: `score = |value − asset_median| / max(1.4826 × MAD, noise_std)`. Answers "you only found it because you had the answer". Structurally weaker, and it **inverts** on long faults — once the fault covers most of the run it drags the median it is measured against until the *healthy* hours look anomalous. `tests/detection/test_statistical.py` pins that failure mode deliberately.
- `AnomalyEvent` / `Incident` / `AnomalyReport` (`incidents.py`) — an event is one deviating (timestamp, asset, parameter) reading; an `Incident` is a contiguous anomalous window plus **ranked candidate assets**, and is pipeline A's Phase 4 hand-off. `Incident.epicenter` is reporting sugar only — Phase 4 should walk `candidates`.
- `AnomalyDetector` (`base.py`) — shared machinery, and where two non-obvious decisions live:
  - **`min_assets` (default 3) controls the false-alarm rate, not the threshold.** A real fault moves 55–95 of Net3's 97 nodes within one timestep, so requiring several assets to disagree simultaneously separates physics from noise without blunting sensitivity to weak faults.
  - **Candidates are interleaved across measurement channels, never sorted by raw score.** Sigma-normalised scores compare *within* a parameter and mislead *across* parameters: flow-meter noise is ~0.001 m³/s against a pressure transducer's 0.30 m, so a trivial flow wobble (23σ) outranks a 3 m pressure collapse (10σ). Sorting on score therefore ranks by *sensor precision* — in a leak scenario the top 20 came back as flow links while the leaking junction, the only asset that can be the answer since a leak is a node fault, sat at rank 21. `_rank_candidates` ranks each channel internally then interleaves, so every channel's best precedes any channel's second. This moved hit@3 from 17% to 71%.
  - Each candidate carries `first_seen_s` (the **first** time that asset disagreed in the incident) and `peak_at_s` (when its kept, strongest reading happened). Until Sept 2026 `first_seen_s` held the peak time — misleading when a sign flips: a pump outage first shows as the pump delivering nothing, while its peak can be the over-run hours later when it refills the tank.
  - Candidates are keyed by **`(parameter, asset_id)`**. EPANET namespaces nodes and links separately, so Net3 has both a junction `101` and a pipe `101`; keying on `asset_id` alone let the louder channel delete the quieter one and could drop the faulted asset from the list entirely.
- `DETECTOR_REGISTRY` maps CLI names (`residual`, `statistical`) to classes, mirroring `FAULT_REGISTRY`.

**Measured results** (`--threshold 4 --min-assets 3`, default noise, 35 scenarios; regenerate with the two Phase 3 commands above):

| run | detected | fault-active recall | precision | false alarms | latency | hit@1 | hit@3 | control |
|---|---|---|---|---|---|---|---|---|
| residual, noise off | 35/35 | 97.6% | 100% | 0% | 0.0h | 17.1% | 57.1% | clean |
| **residual, 1× noise** | **35/35** | **97.6%** | **100%** | **0%** | **0.0h** | **14.3%** | **71.4%** | **clean** |
| residual, 3× noise | 32/35 | 61.4% | 100% | 0% | 0.0h | 14.3% | 28.6% | clean |
| statistical (no twin) | 35/35 | 33.3% | 26.9% | 50.7% | 2.0h | 0.0% | 2.9% | FALSE ALARM |

Reading these:

- **Detection is solved; localization is not.** Every fault is caught, same-hour in 33/35 cases, with zero false alarms — but the faulted asset is top-1 only 14% of the time and median 4 hops away. hit@3 = 71% means Phase 3's honest output is a *shortlist*, and using topology to arbitrate within it is exactly Phase 4's job.
- **Detection skill is not a determinism artifact** — noise-off and 1× noise perform identically. At 3× noise recall degrades gracefully to 61% with precision still 100%; the three that vanish are all 30%-severity leaks, whose signature at the leaking junction (~0.87 m) is comparable to the median per-asset daily pressure swing (~0.88 m).
- **Localization splits by fault type**, because each channel can only name its own kind of asset: blockage hit@1 83% / 0 median hops (a closed pipe's own flow residual is unmistakable), leaks hit@1 0% but hit@5 63% (a leak is a node, so only pressure can name it), pump failure hit@5 100% (the pump's own flow drops, but the downstream links that carried its flow outrank it — a physically sensible confusion the graph can resolve).
- **`demand` is excluded from detection by default.** `LeakFault` *is* an added demand at the target, so the demand residual there equals `severity × max_leak_demand` exactly — detecting a leak that way is circular, and no real network meters demand at every junction.
- **Firing during `recovery` is not scored as a false alarm.** After a fault clears, tank levels and pump schedules have genuinely drifted; baseline deviations in recovery reach 29 m, *larger* than during some faults. The network really is off-nominal, so an alarm is correct — it is simply not the fault window. `evaluate_detection.py` counts false positives on pre-fault hours and the baseline control only, and reports recovery firing (25.4%) separately.
- The **baseline negative control** is the measurement that matters for precision: residuals there are pure sensor noise, so anything that fires is false by construction. `detect_anomalies.py` always runs it.

### Building digital twin (`src/twinrag/building/`)

A generated residential water-supply system that runs through the unchanged pipeline — simulator, faults, graph builder and detectors all consume `Building_G6.inp` exactly as they consume Net3.

- `BuildingSpec` (`spec.py`) — the whole design as data: G+6 massing (3 m storeys, two mirrored 11 x 12 m flats either side of a 4 m service core), the flat floor plan (`DEFAULT_ROOMS`: 10 rooms, 5 wet — kitchen, utility, master bath, common bath, bath 2 — each with a fixture point, demand share and feeding tee), hourly demand patterns (bath/kitchen/utility, normalised to mean 1), 135 L/person/day x 4.5 persons, tank sizes, pump point (2 L/s @ 38 m), level switches (start < 0.6 m, stop > 1.9 m), pipe diameters and minor losses by role, Hazen-Williams C = 140, and **pressure-driven demand** (`PDA`, full flow at >= 3 m).
- `BuildingNetworkGenerator` (`generator.py`) — spec -> `(WaterNetworkModel, layout)`; `write()` saves `.inp` (LPS units) + `.layout.json`. The layout carries what EPANET cannot: true 3D positions, **routed orthogonal pipe paths** (along the slab soffit at +2.7 m, then dropping to fixtures at +0.9 m), and semantics — `floor`, `flat`, `room`, `role`, `label` for every asset — plus the sensor list. The `.inp` coordinates are an oblique projection, so 2D tools still show a building.
- **Asset IDs read as locations and never collide between nodes and links** (unlike Net3's junction `101` / pipe `101`): `F3A-KIT` kitchen point floor 3 flat A (`F0` = ground), `F3A-KIT-BR` its branch pipe, `F3A-IN` flat inlet, `F3A-SUPPLY` flat supply, `F3A-TK`/`F3A-TB` distribution tees, `DTA-3` riser tap-off, `DTA-4-3` riser segment, `OHT`/`SUMP` tanks, `PUMP1`, `CITY` reservoir.
- Topology is a **tree** (130 links, 131 nodes): every link is a cut edge, so any closure starves everything downstream — hence PDA and `pressure_floor_m`.
- **Instrumentation** (`layout["sensors"]`, 103): a flow meter on every room branch (70), flat supply (14), the pump, tank outlet and municipal inlet; pressure at every flat inlet (14) and level in both tanks. Room-branch meters are what make room-level diagnosis possible: within a flat, pipes are a few metres long, so a kitchen leak and a bathroom leak differ by millimetres of pressure — below any transducer — but each moves its own branch's flow.
- Pump controls are EPANET **rules at priority 3**, so a timed outage (priority 6) wins while active and the level switches resume afterwards.
- Baseline physics (pinned in `tests/building/`): pressure falls 3.0 m per storey (ground ~24 m, top-floor taps >= 5.7 m); the roof tank cycles and the pump runs twice a day; a leak shows on its own room branch and flat supply and nowhere else; a closed riser drains the floors below it.

`configs/building.yaml` defines 38 generated scenarios: 30 room leaks (5 rooms spanning both stacks and floors 0-6 x severities 0.1/0.4/1.0 of a 0.2 L/s burst x 2 start hours), 2 ten-hour pump outages (short outages pass unnoticed — the tank is usually full), and 6 blockages at three scales (riser segment `DTA-5-4`, flat supply `F1B-SUPPLY`, kitchen branch `F4A-KIT-BR`).

**Pipeline A on the building, unchanged code** (building sensors only, noise 0.25 m / 0.002 L/s + 2%, `--threshold 4 --min-assets 2`; measured ad hoc, not yet a script): leaks 30/30 detected, the leaking room's branch **hit@1 53%, hit@3 97%**, 0 pre-fault alarms, clean baseline control. When the room is not top-1, `PUMP1` is — the leak drains the roof tank and restarts the pump early. Pump outages detected 2-5 h late (silent until the tank runs down). The riser blockage is detected but `DTA-5-4` is unmetered, so it never appears as a candidate — flats 0A-4A losing pressure together is the evidence, and inferring their common upstream pipe is Phase 4's job. The single-kitchen blockage is missed (one tap's hourly flow is about the meter's noise). `detect_anomalies.py`/`evaluate_detection.py` still hard-code Net3 paths and noise.

### Knowledge graph and retrieval (`graph/knowledge.py`, `retrieval/`) — Phase 4

- `BuildingKnowledgeGraph` (`graph/knowledge.py`) — built from the building `.inp` + layout. Entities: every asset (node *and* link, keyed by its own ID), places (`building`, `floor:3`, `flat:F3A`, `room:F3A-KIT`) and sensors (`sensor:FM-F3A-KIT-BR`). Typed relations: `FEEDS` (node -> pipe -> node, away from the source), `LOCATED_IN` (asset -> most specific place), `PART_OF` (room -> flat -> floor -> building), `MONITORS` (sensor -> asset). 261 assets, 162 places, 103 sensors, 785 relations. Supply direction is **static** because the building is a tree fed from `CITY`; the constructor **rejects loops**, so Net3 cannot be loaded. Queries: `upstream(asset)` (supply path to source), `downstream`, `common_supply_point(assets)` (deepest shared supplier), `neighbourhood(asset, hops)` (hops = pipes, direction ignored), `triples(entities)`, `places_of`.
- `SubgraphRetriever` (`retrieval/retriever.py`) — `retrieve(incident) -> EvidencePacket`. Physics, not plain BFS, decides what matters:
  - **Extra flow** (leak) -> the *deepest* alarmed pipe carrying flow above the twin, plus the tap it feeds. Water escaping is drawn through every pipe above it, so only the deepest is specific.
  - **Pressure loss** (closure) -> the common supply point of the local pressure-loss alarms, then walk **up** until a node that also feeds a healthy pressure sensor: the fault is in that suspect span (riser blockage -> `[DTA-4, DTA-5-4]`, bounded by `DTA-5` because flat 5A is fine).
  - **Building-wide shift** -> when most flats' pressure moves together (roof tank low), those alarms are summarised as one fact and removed from localisation; a flat standing out from the median shift by >= `local_sigma` (4) of its noise stays local. Without this, a big leak's packet grew to ~20k tokens and pointed at the roof manifold.
  - **Tank levels are their own signal** -> a low tank is traced up its supply to the first metered asset (`OHT -> RISING-MAIN -> PUMP-DEL -> PUMP1`); mixing it into line pressure dragged the common point to the roof.
  - **Scale matters**: pressure losses smaller than `minor_fraction` (0.25) of the largest are reported (`minor_pressure_loss_not_used_for_localisation`) but not used to locate the fault — a dry flat (-20 m) and a knock-on wobble elsewhere (-1.2 m) share only the roof as a common supplier.
  - **Time order**: pass `events=report.events` and every alarm gets its hourly signed deviation (`hourly_deviation`, true `first_seen`, peak time), readings from up to `lookback_hours` (3) *before* the incident opened are kept as early alarms (lone readings the `min_assets` rule could not open an incident on — e.g. the pump delivering nothing at 11:00), and `incident.sequence_of_first_alarms` lists when each asset first went off-twin.
  - Context = anchors' supply paths to the common anchor and to the source, a `hops`-pipe neighbourhood, **the whole flat of any in-flat anchor** (sibling rooms sit on the other tee, beyond a short BFS, and "the other rooms are fine" is what pins a room), and the sensors in it that did *not* alarm (negative evidence, capped at 30).
- `EvidencePacket` (`retrieval/evidence.py`) — `incident` (times only), `building`, `observations` (alarms with readings in L/s or m and noise-scaled score; one aggregated row for a building-wide shift; normal sensors), `assets` (id, kind, role, label, location, metered), `relations` (triples), `topology_facts`, `allowed_ids` (Phase 5 must reject any diagnosis citing an ID outside it). `assert_no_leakage(forbidden=[scenario])` fails if the serialised packet contains the scenario name or ground-truth words (`fault_active`, `recovery`, `scenario`, `severity`, `injected`); `build_evidence_packets.py` calls it on every packet.
- **Results** (`build_evidence_packets.py`, defaults: threshold 4, min-assets 2, hops 2): baseline control 0 incidents; detected 36/38 (both misses are the single-kitchen blockage); **the true faulty asset is in the packet and inside the topology "focus" for 36/36 detected scenarios** (30 leaks, 2 pump outages, 4 blockages); median packet ~40 assets / ~4.2k tokens (range ~2.4k-6.7k). The focus is a shortlist (usually 2 assets, up to ~6 when a big leak also drains the roof tank or a flat's pressure wobbles), so choosing within it is Phase 5's job. Caveat: the retrieval rules were designed while looking at these 38 scenarios — validate on fresh targets/severities before quoting 100% in the paper.

### LLM reasoning (`src/twinrag/reasoning/`) — Phase 5

- `ChatClient` (`llm.py`) — standard-library HTTP client for any OpenAI-compatible `/chat/completions`; no SDK dependency. Settings from env or the gitignored project `.env`: `TWINRAG_LLM_BASE_URL`, `TWINRAG_LLM_API_KEY` (falls back to `HF_TOKEN`), `TWINRAG_LLM_MODEL`; defaults are the Hugging Face router + Llama-3.3-70B. Temperature 0, `max_tokens` 4000 (reasoning models such as gpt-oss spend output tokens thinking first). Asks for `response_format: json_schema` and retries as plain text only on HTTP 400/422. HTTP 429 waits for the server's `retry-after` (own budget of 8 waits); other 4xx fail immediately. Sends an explicit `User-Agent` — the HF gateway's firewall answered Python's default urllib agent with an HTML 403.
- Provider notes (Sept 2026): **Hugging Face** free accounts get ~$0.10/month of credit and this account's was already exhausted (HTTP 402) before the first real run. **Groq** free tier (`https://api.groq.com/openai/v1`, keys start `gsk_`) is what runs now: `openai/gpt-oss-120b` (open weights) or `qwen/qwen3.8-27b`; Llama 3.3 70B is enterprise-only there. Limits ~8k tokens/min and ~200k tokens/day per model — one ~5k-token packet a minute, so pass `--delay 45` and expect a full 36-incident run to take ~45-60 min and a large share of a day's budget.
- `render_packet` (`prompt.py`) — the LLM gets a compact text rendering, not raw JSON: SEQUENCE, ALARMS (with hourly deviations), normal sensors as one list, one line per asset, FEEDS as `A->B`, topology facts as compact JSON. ~1.8-3.2k tokens instead of up to ~9.9k — needed because Groq's free tier counts prompt + `max_tokens` against ~8k tokens/min and answered HTTP 413 otherwise.
- `SYSTEM_PROMPT` (`prompt.py`) — domain knowledge only (how the building is supplied, how IDs read, what each topology fact means, that knock-on effects must be separated from the cause) plus hard rules: use only the packet, cite only `allowed_ids`, one root-cause asset, 2-6 reasoning steps each citing IDs, JSON matching `DIAGNOSIS_SCHEMA` (`root_cause_asset`, `fault_type` in leak/blockage/pump_failure/other, `location`, `affected_assets`, `reasoning[{step, evidence}]`, `alternatives`, `confidence`, `recommended_action`). Nothing scenario-specific is ever in the prompt.
- `Diagnoser` (`diagnosis.py`) — calls the model, parses JSON (tolerates fences/prose), runs `check_grounding` (every cited ID in `allowed_ids`, valid type, confidence in [0, 1]); on failure sends the exact problems back once (`repair_attempts=1`). A still-failing answer is kept and marked `grounded=False`.
- `ungrounded_packet` — the **no-graph ablation**: only alarmed sensors with asset, label and reading; no relations, topology facts, normal sensors or asset list. Run with `repair_attempts=0` so citing an ID it was never shown counts as hallucination instead of being repaired away. Caveat: building IDs are human-readable, so a model can *guess* `DTA-5-4` from the naming pattern; that shows up as an ungrounded citation.
- Known issue: models sometimes cite sensors as `FM-PUMP1` instead of `sensor:FM-PUMP1`, which `check_grounding` rejects; a repair turn can then exceed the free tier's request size (HTTP 413). Normalise the prefix before tightening anything else.
- `diagnose_incidents.py` scores each diagnosis after the fact: `exact` (named the faulted asset), `room` (exact or same room — a leaking tap vs its branch pipe), `near` (same room, or one pipe away — tap vs its branch, riser segment vs the node below), `type_correct`, `grounded`; alongside the detector's own top-ranked suspect (`detector_top1` from `retrieval_summary.csv`) as the baseline. Answer key is read only for scoring.

### Configuration (`configs/simulation.yaml`, `configs/building.yaml`)

Two independent ways to define scenarios, both read from the same file:

- `faults:` — an explicit hand-written list, consumed by `run_fault_simulation.py`.
- `scenario_generation:` — the combinatorial matrix (`enabled`, then a `leak`/`pump_failure`/`blockage` block each with `target_ids`, `severities`, `start_hours`, `duration_hours`), consumed by the `*_generated_*` scripts.

The two do not interact — `run_fault_simulation.py` ignores `scenario_generation`, and the generated-batch scripts ignore `faults`. `run_generated_scenarios.py` takes `--config`, `--output-root` and `--with-baseline`; with no flags it reproduces the Net3 batch into `data/generated/` exactly as before.

### Data layout

- `data/networks/*.inp` — input EPANET models (committed; e.g. `Net3.inp`). `Building_G6.inp` + `Building_G6.layout.json` are generated by `build_building_network.py`.
- `data/building/` — the building batch: `processed/` (incl. `baseline.csv`), `metadata/`, `scenarios_manifest.csv` (all gitignored), and the viewer build: `building_twin.html` is committed (~2.7 MB, three.js inlined) so the twin opens without a Python setup; its intermediate payload `building_view.json` is gitignored.
- `data/processed/` — datasets from the baseline/explicit-fault runs (gitignored except `.gitkeep`).
- `data/metadata/` — per-scenario ground-truth fault labels as `<scenario>.json` (gitignored), written alongside each fault dataset by `run_fault_simulation.py`.
- `data/building/diagnosis/<run>/` — Phase 5 output: per-scenario diagnosis + score JSON, `summary.csv`, `run.json` (model, token usage). Gitignored.
- `data/building/evidence/` — Phase 4 output: one evidence packet per scenario (`<scenario>.json`; the filename is for humans, the content never names the scenario) and `retrieval_summary.csv`. Gitignored.
- `data/generated/processed/`, `data/generated/metadata/` — the generated evaluation batch, same file naming, written by `run_generated_scenarios.py` (gitignored).
- `data/generated/scenarios_manifest.csv` — index of the batch, one row per scenario: `scenario_id, scenario, fault_type, target_id, severity, start_hour, end_hour, dataset_file, metadata_file, rows`. `validate_generated_dataset.py` and `detect_anomalies.py` both drive entirely off this file, so it is the entry point for consuming the batch. Note it is written by `csv` on Windows and therefore carries **backslash** path separators — normalise them (`str.replace("\\", "/")`) when resolving, as the Phase 3 scripts do.
- `data/generated/detection/<run>/` — pipeline A output, one directory per detection run (`summary.csv`, `incidents.json`, `run.json`, `evaluation.csv`, `evaluation.json`, and `events/<scenario>.csv`). Gitignored.
- `data/detection/` and `data/evaluation/` — pipeline B output. **These are tracked in git**, unlike every other generated artifact in the repo. Left as-is because they arrived that way; untracking them (`git rm --cached`) is a team decision, not a mechanical one.
- `data/generated/graph_view.json`, `twin_viewer.html`, `twin_viewer.fragment.html` — viewer build artifacts. These are currently **tracked in git**, unlike everything else under `data/generated/`.

Generated simulation artifacts (`*.bin`, `*.hyd`, `*.rpt`, and the contents of `data/processed`, `data/metadata`, `data/generated/{processed,metadata,detection}`) are gitignored — do not commit them.

### Viewer (`viewer/template.html`) — static Phase 6 prototype

A hand-rolled SVG network map with a time scrubber, colour-by-pressure or colour-by-deviation-vs-normal, and a node inspector. No CDN, no framework, no backend: `export_graph_view.py` bakes the topology plus **4 hardcoded scenarios** (baseline + one of each fault type, listed in that script's `SCENARIOS`) into a JSON payload, and `build_viewer.py` inlines it into both a standalone document and an embeddable fragment. Theme-aware, percentile-clamped colour domains.

It predates Phases 3–5, so it shows no anomalies, no retrieved subgraph and no diagnosis. Treat it as a rendering layer worth reusing rather than a finished dashboard. `server_viewer.py`'s docstring still calls itself `serve_viewer.py` from before the rename.

### Building viewer (`viewer/building_template.html`) — 3D Phase 6 prototype

three.js r147 (UMD build + `OrbitControls`, vendored in `viewer/vendor/` with its MIT licence) inlined by `build_building_viewer.py` into one offline HTML file carrying all 39 runs. Kept deliberately simple for non-specialist viewers: slabs, corner/core columns and room outlines (no walls), pipes along their routed paths, taps, tanks with live water level. Two colour modes: **Problems** (default) classifies every asset against the twin's normal day at the same hour -- normal (muted), *changed* at >= 3x instrument noise, *problem* at >= 8x, *no water* when a normally pressurised point drops below 0.3 m -- and enlarges abnormal assets so they stand out; **Pressure** is a sequential ramp. The side panel summarises sensor-only alerts in plain language ("Flat 3A supply -- flow up 0.200 L/s") and names the most affected flats. **Ground truth is hidden by default**: no injected-fault marker, and scenario names read "Leak test 17" rather than the location, until *More options -> Reveal answer* is ticked -- the colours, not the label, should point to the problem. The payload carries **pressure and flow only, never node `demand`**: EPANET's node demand is everything leaving the pipes there, and a `LeakFault` *is* extra water leaving at the target, so demand at the leaking tap reads ~80x normal (an early build showed "8539% of normal water reaching the tap" and leaked the answer). "Tap can deliver" is instead computed from pressure with EPANET's own PDA rule (`demand_model`, read from the `.inp`: 0 at minimum pressure, 100% at required pressure, square-root between), so it cannot exceed 100%. Picking is screen-space nearest-sample, not raycasting (thin pipes are hard to hit). Deep links: `building_twin.html#scenario=<name>&floor=3&sel=F3A-KIT&t=10&mode=pressure&spread=1&answer=1`. Headless check: Chrome with `--use-angle=swiftshader --enable-unsafe-swiftshader --screenshot` renders it; note headless Chrome on Windows lays out at >= 504 px regardless of `--window-size`.

### Tests

`tests/` mirrors the package layout: `tests/simulation/`, `tests/graph/`, `tests/detection/`, `tests/building/`, `tests/retrieval/`, `tests/reasoning/`. **146 tests, all passing** (19 of them skip until `data/processed/baseline.csv` exists). Only detection pipeline A is covered; pipeline B has no tests.

- `tests/reasoning/` — no network: a scripted `FakeClient` stands in for the LLM. Grounded answers pass; invented IDs, bad fault type or confidence are violations; JSON extraction from fences/prose; the repair turn fixes an ungrounded answer and tells the model what was wrong; an unrepaired answer is kept but marked; the no-graph input carries alarms only.
- `tests/retrieval/` — simulation-free. `test_knowledge_graph.py` (entity/relation counts, tap-to-street supply path, downstream of a riser segment, common supply point, typed triples, hop counting, loop rejection); `test_retriever.py` (hand-built `Incident`s: extra flow -> deepest branch + tap with sibling rooms as negative evidence, closed riser bracketed to `[DTA-4, DTA-5-4]` by healthy flat 5A, uniform drop -> one building-wide fact + low-tank chain to `PUMP1`, a flat standing out from the shift stays local, every cited ID is in `allowed_ids`, no ground truth in the packet).

- `tests/building/` — composition, ID uniqueness, layout covers every asset, fixtures sit inside their rooms, pipe paths join their endpoints and are orthogonal, the topology is a tree, flat B mirrors flat A, every room branch is metered, `.inp` round-trip keeps rules and PDA; then real EPANET runs (sub-second) for the physics listed under *Building digital twin*. Runs EPANET in a temp cwd so `temp.inp` is not touched.

- `tests/simulation/` — `test_network_loader.py` (Net3 loads; exact component counts — 92 junctions / 2 reservoirs / 3 tanks / 117 pipes / 2 pumps; missing file raises), `test_scenario_generator.py` (generated counts, valid time windows, unique scenario names), `test_scenario_labels.py` (the `normal`/`fault_active`/`recovery` boundaries, using a `SimpleNamespace` stub fault and a hand-built dataframe rather than a real simulation). Tests call the runner's private `_add_*_state_labels` helpers directly to stay fast — keep those names stable or update the tests.
- `tests/graph/` — `test_builder.py` (composition matches the network, one connected component, reservoir head → `elevation`), `test_snapshot.py` (values attach, flow direction follows sign, link 105 genuinely reverses between hour 4 and hour 16, neighbour ordering, upstream reaches a source), `test_validation.py` (the `GraphValidator` checks). These **run a real EPANET load** and read `data/processed/baseline.csv`, skipping if it is absent.
- `tests/detection/` — pipeline A only; fast and simulation-free, synthetic frames with no EPANET. `test_sensors.py` (seed reproducibility, row-order independence, truth preserved, coverage filtering), `test_residual.py` (identical runs are silent, `min_assets` suppresses an isolated spike, score really is in sigma, separate windows stay separate incidents, node/link ID collision, channel interleaving), `test_statistical.py` (needs no baseline, constant assets are not flagged, and the long-fault inversion), `test_incidents.py` (the serialization contract Phase 4 consumes — named to match `incidents.py`, leaving `test_events.py` free for pipeline B).
