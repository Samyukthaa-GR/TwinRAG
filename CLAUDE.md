# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TwinRAG is a digital-twin platform for water distribution networks. The end goal: an anomaly fires somewhere in the network, a subgraph around that point is extracted from the knowledge graph, and that local physical context is handed to an LLM which reasons out the root cause with no manual pipe-tracing. The simulator exists to manufacture ground truth for that pipeline.

Six planned phases. **Phases 1–3 are built:**

1. **Simulation** — EPANET/`wntr` runs on `.inp` models, timed fault injection, long-format datasets. Complete.
2. **Knowledge graph** — `networkx` topology, per-timestep hydraulic snapshots with flow direction and neighbour lookup, plus a schema/validation layer. Complete.
3. **Anomaly detection** — **two parallel implementations, both kept.** See *Anomaly detection* below; this is the single most important thing to understand before touching the package.
4. **Topology-aware retrieval** — BFS subgraph around an incident, pruned by hydraulic relevance, serialized as an LLM evidence packet. **Not built.**
5. **LLM root-cause reasoning** — structured diagnosis from the evidence packet, scored against the metadata answer key. **Not built.** No LLM client in `requirements.txt` yet.
6. **Operator dashboard** — a static prototype viewer exists (see *Viewer* below); the live backend, anomaly overlay and diagnosis panel do not.

> **Two Phase 3 pipelines coexist in `src/twinrag/detection/`** — a dataclass/object pipeline and a DataFrame/CSV pipeline, built independently by two contributors and merged deliberately rather than consolidated. They share no types. Pick one per consumer; do not interleave them. **Phase 4 must choose one contract** — `Incident` (dataclass) or the `anomaly_events.csv` rows (DataFrame) — and that decision is still open.

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

# Tests (pytest.ini sets pythonpath=src and testpaths=tests — no sys.path juggling needed)
pytest
pytest tests/detection                                                             # one package
pytest tests/simulation/test_scenario_generator.py::test_generated_scenario_count   # single test
```

On Windows use the `py` launcher and the venv interpreter explicitly: `py -m venv .venv`, then `.\.venv\Scripts\python.exe ...`. The bare `py`/`python` on PATH does *not* have the dependencies installed. The full suite (57 tests) and real EPANET simulations both run in this environment; the simulation-dependent graph tests skip if `data/processed/baseline.csv` has not been generated.

Note: the machine's Application Control policy intermittently blocks pandas' compiled DLLs on first touch — `ImportError: DLL load failed while importing <ext>: An Application Control policy has blocked this file`. It has cleared as of the latest check (pandas 3.0.5 imports, the suite is green, `run_baseline_simulation.py` completes). If it reappears, retrying the same command usually succeeds; it is an OS-level policy quirk, not a code or dependency problem.

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

### The graph layer (`src/twinrag/graph/`) — Phase 2

- `NetworkGraphBuilder` (`builder.py`) — turns a `WaterNetworkModel` into a `networkx` graph. Junctions/tanks/reservoirs become nodes (reservoir `base_head` is normalised onto the same `elevation` field as junction elevation); pipes/pumps/valves become edges carrying `link_name`, `start_node`, `end_node`, diameter, length. `summary(graph)` gives composition + connectivity for sanity checks.
- The graph is deliberately **undirected**: 56 of Net3's 119 links reverse over a normal 24h day as tanks fill and drain, so direction is a property of a moment, not of the network.
- **Known limitation:** `add_edge(..., key=name)` on an `nx.Graph` stores `key` as an ordinary attribute — it is *not* a MultiGraph key. Parallel links between the same node pair would silently overwrite each other. Net3 has none (119 links → 119 edges), so this is latent, but the docstring's promise is not currently kept. Switch to `nx.MultiGraph` before trusting it on another network.
- `GraphSnapshot` (`snapshot.py`) — lays one timestep over the topology: pressure/demand onto nodes, flowrate onto links, and `flow_from`/`flow_to` derived from the **sign** of each flowrate. `directed()` returns the flow-oriented `DiGraph` (zero-flow links omitted); `neighbors(asset_id, hops)` is the BFS retrieval primitive Phase 4 builds on; `upstream()`/`downstream()` give causal ancestry and blast radius at that instant.

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
- `data/generated/scenarios_manifest.csv` — index of the batch, one row per scenario: `scenario_id, scenario, fault_type, target_id, severity, start_hour, end_hour, dataset_file, metadata_file, rows`. `validate_generated_dataset.py` and `detect_anomalies.py` both drive entirely off this file, so it is the entry point for consuming the batch. Note it is written by `csv` on Windows and therefore carries **backslash** path separators — normalise them (`str.replace("\\", "/")`) when resolving, as the Phase 3 scripts do.
- `data/generated/detection/<run>/` — pipeline A output, one directory per detection run (`summary.csv`, `incidents.json`, `run.json`, `evaluation.csv`, `evaluation.json`, and `events/<scenario>.csv`). Gitignored.
- `data/detection/` and `data/evaluation/` — pipeline B output. **These are tracked in git**, unlike every other generated artifact in the repo. Left as-is because they arrived that way; untracking them (`git rm --cached`) is a team decision, not a mechanical one.
- `data/generated/graph_view.json`, `twin_viewer.html`, `twin_viewer.fragment.html` — viewer build artifacts. These are currently **tracked in git**, unlike everything else under `data/generated/`.

Generated simulation artifacts (`*.bin`, `*.hyd`, `*.rpt`, and the contents of `data/processed`, `data/metadata`, `data/generated/{processed,metadata,detection}`) are gitignored — do not commit them.

### Viewer (`viewer/template.html`) — static Phase 6 prototype

A hand-rolled SVG network map with a time scrubber, colour-by-pressure or colour-by-deviation-vs-normal, and a node inspector. No CDN, no framework, no backend: `export_graph_view.py` bakes the topology plus **4 hardcoded scenarios** (baseline + one of each fault type, listed in that script's `SCENARIOS`) into a JSON payload, and `build_viewer.py` inlines it into both a standalone document and an embeddable fragment. Theme-aware, percentile-clamped colour domains.

It predates Phases 3–5, so it shows no anomalies, no retrieved subgraph and no diagnosis. Treat it as a rendering layer worth reusing rather than a finished dashboard. `server_viewer.py`'s docstring still calls itself `serve_viewer.py` from before the rename.

### Tests

`tests/` mirrors the package layout: `tests/simulation/`, `tests/graph/`, `tests/detection/`. **105 tests, all passing.** Only detection pipeline A is covered; pipeline B has no tests.

- `tests/simulation/` — `test_network_loader.py` (Net3 loads; exact component counts — 92 junctions / 2 reservoirs / 3 tanks / 117 pipes / 2 pumps; missing file raises), `test_scenario_generator.py` (generated counts, valid time windows, unique scenario names), `test_scenario_labels.py` (the `normal`/`fault_active`/`recovery` boundaries, using a `SimpleNamespace` stub fault and a hand-built dataframe rather than a real simulation). Tests call the runner's private `_add_*_state_labels` helpers directly to stay fast — keep those names stable or update the tests.
- `tests/graph/` — `test_builder.py` (composition matches the network, one connected component, reservoir head → `elevation`), `test_snapshot.py` (values attach, flow direction follows sign, link 105 genuinely reverses between hour 4 and hour 16, neighbour ordering, upstream reaches a source), `test_validation.py` (the `GraphValidator` checks). These **run a real EPANET load** and read `data/processed/baseline.csv`, skipping if it is absent.
- `tests/detection/` — pipeline A only; fast and simulation-free, synthetic frames with no EPANET. `test_sensors.py` (seed reproducibility, row-order independence, truth preserved, coverage filtering), `test_residual.py` (identical runs are silent, `min_assets` suppresses an isolated spike, score really is in sigma, separate windows stay separate incidents, node/link ID collision, channel interleaving), `test_statistical.py` (needs no baseline, constant assets are not flagged, and the long-fault inversion), `test_incidents.py` (the serialization contract Phase 4 consumes — named to match `incidents.py`, leaving `test_events.py` free for pipeline B).
