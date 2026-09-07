# rdspd-core — Synthetic Retail Credit Portfolio Simulator

[![CI](https://github.com/Atheia/rdspd-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Atheia/rdspd-core/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

`rdspd-core` generates a **synthetic portfolio of revolving consumer credit contracts** and
simulates its life month by month over ~18 years: risk-driver updates, revolving drawdowns,
arrears and defaults, repayment schedules, and contract closure. It applies **behavioural
scorecards** (drawdown propensity and probability of default) and ships a suite of
**statistical validation controls** with confidence intervals.

It is a production-grade port of a research notebook into a modular, tested Python package.
The business logic is fully decoupled from I/O and the CLI, so it can be driven from a script,
a notebook, or a future API layer.

> **Why it exists.** Realistic retail-credit datasets are rarely shareable. This simulator
> produces a fully synthetic but economically coherent portfolio — calibrated on macro series
> (unemployment, inflation, income indices, interest rates) — suitable for model development,
> backtesting, IFRS 9 / Basel experimentation, and teaching.

---

## Highlights

- **Reproducible** — a global seed makes every run deterministic.
- **Validated** — the deterministic scorecard grids are reproduced **bit-for-bit** against
  reference files, and the model ships built-in validation controls (covariance, conditional
  distributions, 95% confidence intervals on P0/P1/P2, PDt and 12-month PD).
- **Observable** — every major stage is timed and reported in the terminal and on disk.
- **Fast** — heavily vectorised: the full pipeline runs in seconds on a reduced portfolio and
  a few minutes at 50 000 contracts × 216 monthly snapshots.
- **Ready to run** — model parameters are included; `pip install -e .` then one command.

---

## Quick start

Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/Atheia/rdspd-core.git
cd rdspd-core
python -m venv .venv && .venv\Scripts\activate      # Windows
# python -m venv .venv && source .venv/bin/activate # Linux / macOS

pip install -e ".[dev]"
rdspd-simulate --smoke
```

That's it — the model parameters ship with the repository, so the simulation runs immediately.

```
======================================================================
  Temps d'execution par etape
======================================================================
  Octroi des contrats              355 ms  (  7.8 %)
  Grilles de score                 467 ms  ( 10.2 %)
  Initialisation                    71 ms  (  1.6 %)
  Simulation mensuelle             2.47 s  ( 64.5 %)
  Consolidation                    387 ms  (  8.5 %)
  Controles de validation          174 ms  (  3.8 %)
  Exports finaux                    86 ms  (  1.9 %)
  Nettoyage des pickles              5 ms  (  0.1 %)
  TOTAL                            3.83 s  (100.0 %)
```

Run the full portfolio (50 000 contracts, 2006 → 2023) with `rdspd-simulate`.

---

## What the simulation does

| Stage | Description | Main output |
|------|-------------|-------------|
| **Origination** | Generates correlated, non-normal borrower attributes (Pearson distribution system), assigns vintages, de-standardises, draws categorical variables (occupation, housing, marital status, seniority) and dates. | `00_Granting.csv` |
| **Macro factors** | Loads unemployment rates and durations, inflation, income indices per occupation, revolving interest rates, and an aggregate macro factor. | — |
| **Scorecards** | Builds the revolving (RVV) and default (PD) score grids from the Cartesian product of risk-driver modalities; optimises the decomposition `PDt = P0 × P1 × P2`. | `ScoreGrid_RVV.csv`, `ScoreGrid_PD.csv` |
| **Monthly simulation** | 216 monthly snapshots: risk-driver updates, drawdown draws, arrears and defaults conditional on P0/P1/P2 and asset correlation ρ, repayment schedule, closure. | monthly snapshots |
| **Consolidation & controls** | Concatenates snapshots, computes the 12-month default flag, runs the validation controls. | `controls/` |
| **Exports** | Full dataset and the performing December snapshot. | `RDSPD_Full_<version>.parquet`, `RDSPD_Perf_Dec_<version>.parquet` |

### Modelling in brief

Default is modelled as **three consecutive months of arrears**. Each month, the probability of
falling into (or staying in) arrears is drawn from the scorecard-implied `P0`, `P1`, `P2`
(first, second, third consecutive month). The one-factor **asset correlation** parameter `ρ`
couples `P0` to the aggregate macro factor, so defaults cluster in downturns.

---

## Usage

```bash
rdspd-simulate                                        # full run
rdspd-simulate --smoke                                # reduced run (fast)
rdspd-simulate --n-contracts 5000 --end 2010-12-31    # custom
rdspd-simulate --exportformat csv                     # CSV instead of Parquet
```

| Option | Default | Description |
|--------|---------|-------------|
| `--smoke` | — | Reduced preset (2 000 contracts, 2006 → 2007) |
| `--version` | `PD02R50k0623TTC` | Run identifier (= output folder name) |
| `--n-contracts` | `50000` | Portfolio size |
| `--start` / `--end` | `2006-01-31` / `2023-12-31` | Snapshot range (month ends) |
| `--rho` | `0.05` | Asset correlation (impact on `P0`) |
| `--seed` | `42` | Global random seed (reproducibility) |
| `--exportformat` | `parquet` | Final export format: `parquet` or `csv` |
| `--inputs-dir` / `--outputs-dir` | `inputs/parameters` / `outputs/` | I/O directories |
| `--scorecard-pd` / `--scorecard-rvv` | `PD02R3120` / `RVV00R0030` | Scorecards used |
| `--no-clean` | — | Do not purge `outputs/<version>/` before the run |

### Programmatic use (API-ready)

```python
from rdspd_core.config import SimulationConfig
from rdspd_core.pipeline import run

result = run(SimulationConfig(n_contracts=5000, end="2010-12-31"))
print(result.full_path, result.n_rows_full, result.total_seconds)
```

`pipeline.run(config)` is the single entry point: no hard-coded paths, no global state beyond
pandas display options — designed to be wrapped by a web API.

---

## Outputs

Everything is **printed during execution** *and* written to `outputs/<version>/`, which is
purged at the start of each run (use `--no-clean` to keep existing files):

```
outputs/<version>/
    00_Granting.csv                      # originated portfolio (re-read internally)
    ScoreGrid_RVV.csv                    # revolving score grid
    ScoreGrid_PD.csv                     # default score grid
    RDSPD_Full_<version>.parquet         # full dataset (or .csv)
    RDSPD_Perf_Dec_<version>.parquet     # performing December snapshot (or .csv)
    run.log                              # execution log
    controls/                            # one CSV per validation control
    controls/timings.csv                 # duration of each stage
```

---

## Reproducibility & validation

- **Deterministic grids.** `ScoreGrid_PD.csv` (3 750 × 20) and `ScoreGrid_RVV.csv` (32 × 11)
  are produced by a pure Cartesian product plus a deterministic optimisation. They are checked
  **bit-for-bit** against the reference files in `inputs/reference/` (see
  `tests/test_scores_deterministic.py`).
- **Seeded runs.** With a fixed `--seed`, two runs of the same configuration are identical.
- **Built-in controls.** Covariance error, variable distributions, age-conditional categorical
  distributions, risk-driver → score mapping, and 95% confidence intervals on P0/P1/P2, PDt and
  the 12-month PD. Failures are printed and written to `controls/`.
- **Encoding.** Several parameter files are cp1252-encoded and the PD scorecard matches accented
  labels literally (e.g. `Indépendants`). Reading is encoding-robust and covered by a test.

---

## Architecture

```
src/rdspd_core/
    config.py        # SimulationConfig: all parameters, no hard-coded paths
    io_utils.py      # encoding-robust CSV reading, Parquet/CSV export
    reporting.py     # console output + files + per-stage timing
    randnonnormal/   # non-normal sampling (Pearson distribution system)
    granting.py      # contract origination
    macro.py         # macro-economic factors
    risk_drivers.py  # monthly risk-driver updates
    scores.py        # RVV/PD grids + score application
    snapshots.py     # contract/facility lifecycle
    simulation.py    # monthly loop
    controls.py      # validation controls
    pipeline.py      # end-to-end orchestration (API entry point)
    cli.py           # command-line interface
```

Inputs live in `inputs/parameters/` (model calibration, versioned) and `inputs/reference/`
(reference grids used by the tests).

---

## Tests

```bash
pytest
```

- `test_scores_deterministic.py` — generated grids match the reference bit-for-bit.
- `test_encoding.py` — accented labels decode correctly.
- `test_smoke_pipeline.py` — end-to-end reduced run, export formats, output cleaning,
  reproducibility.

---

## Performance

The pipeline is vectorised throughout; every optimisation was validated **byte-for-byte**
against reference outputs before being merged:

- vectorised origination (~17×),
- lighter monthly loop (merges → lookups, no pickle re-reads),
- vectorised 12-month PD control (~46×),
- `object` rather than Arrow string storage (`future.infer_string=False`, ~25% on the loop with
  pandas ≥ 3.0),
- Parquet exports (~9× faster to write than CSV) and removal of redundant exports.

Each stage is timed, so regressions are visible immediately (`controls/timings.csv`).

---

## License

Released under the [MIT License](LICENSE).

The non-normal sampling module implements published mathematics (Pearson distribution
system; Heinrich 2004 for the type IV sampler; Vale & Maurelli 1983 for the
independent-generator construction) — see [NOTICE](NOTICE) for the full references.
