# Input data

Everything the simulation reads lives here. Two folders, two very different purposes:

| Folder | Content | Safe to edit? |
|--------|---------|---------------|
| `parameters/` | Model calibration — 17 small CSV files read at every run | **Yes**, that is the point |
| `reference/` | Score grids the test suite compares against, bit-for-bit | No, unless you regenerate them on purpose |

The folder is selected with `--inputs-dir` (CLI) or `inputs_dir=` / `project_root=`
(`SimulationConfig`), so you can keep your own calibration outside the repository.

---

## `parameters/` — what each file drives

### Origination (portfolio generation)

| File | Role | Structure |
|------|------|-----------|
| `RDSPD_CovarMatrix_Conso.csv` | Target correlation matrix between the six borrower variables (age, children, income, maturity, charges, credit amount). Drives the non-normal correlated draw. | 6 × 6, symmetric, `1` on the diagonal |
| `RDSPD_Quantile_Conso.csv` | Quantiles per variable and per year. Used both to de-standardise the simulated variables and to derive the skewness/kurtosis of the Pearson distributions (from `reference_year`, default 2021). | long: `variable`, `Year`, `Q10`, `Q25`, `Q50`, `Q75`, `Q90` |
| `RDSPD_Cohort_Revolving.csv` | Relative weight of each origination vintage. Only **2006 → 2023** is used; weights are renormalised internally, so they need not sum to anything in particular. | `Year`, `Weight` |

### Categorical attributes, conditional on age class

All four share the same shape: one row per age class (`<25`, `25-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70-79`, `>=80`), one column
per modality, each **row summing to 1**.

| File | Modalities |
|------|-----------|
| `RDSPD_ProbaCSP_Conso.csv` | Occupation: `Agriculteurs`, `Indépendants`, `Cadres`, `Professions intermédiaires`, `Employés`, `Ouvriers` |
| `RDSPD_ProbaHousing_Conso.csv` | `Propriétaire` / `Non Propriétaire` |
| `RDSPD_ProbaFamily_Conso.csv` | `Couple` / `Single` |
| `RDSPD_ProbaEmploy_Conso.csv` | Employment seniority: `<1`, `1-5`, `5-10`, `>= 10` |

### Macro-economic series

| File | Role | Structure |
|------|------|-----------|
| `RDSPD_Unemployment.csv` | Quarterly probability that an employed borrower becomes unemployed. | `Date` (`DD/MM/YYYY`), `Mu_t` |
| `RDSPD_Unemployment_Duration.csv` | Distribution of the unemployment spell, in months. | `Duration`, `Proba` (sums to 1) |
| `RDSPD_Inflation.csv` | Inflation index, base 2000 = 1.00. Deflates income and charges. | `Year`, `Base` |
| `RDSPD_IncomeIndex.csv` | Income index per occupation and per year, base 1996 = 100. | `Annee` + one column per occupation |
| `RDSPD_InterestRate_Revolving.csv` | Average monthly revolving rate applied to outstanding balances. | `Date` (`YYYY/MM/DD`), `InterestRateAvg` |
| `RDSPD_AggMacro.csv` | Quarterly aggregate macro factor. Coupled to `P0` through the asset correlation `--rho`, this is what makes defaults cluster in downturns. | `Date` (`DD/MM/YYYY`), `AggMacro` |

### Scorecards

Two files per scorecard, named after its identifier (`PD02R3120` for default, `RVV00R0030`
for drawdown propensity — selected with `--scorecard-pd` / `--scorecard-rvv`).

| File | Role | Structure |
|------|------|-----------|
| `RDSPD_PartialScore_<id>.csv` | Partial score of every risk-driver modality. The **first row must be the `Intercept`** (read by position). | `Var`, `Mod`, `PartialScore` |
| `RDSPD_ScoreParams_<id>.csv` | Score → probability transform: `Odds = OddsRatio ^ ((Score - ScoreIso) / ScoreAmpli)`. The **three rows must stay in this order** (read by position). | `ScoreAmpli`, `OddsRatio`, `ScoreIso` |

Risk drivers used by the PD scorecard: `rd_Age_AncEmp`, `rd_DebtRatioW`, `situation_habitat`,
`rd_UsingRate`, `rd_CSP`, `rd_BankingSen` — by the RVV scorecard: `dec_flag`,
`dispo_ratio_bin`, `draw_senior_bin`.

---

## Rules to follow when editing

1. **Probabilities must sum to 1** on every row of the `RDSPD_Proba*` files and over
   `RDSPD_Unemployment_Duration.csv`. Nothing renormalises them for you.
2. **Keep the encoding.** Several files are **cp1252**-encoded and the PD scorecard matches
   accented labels *literally*: `Indépendants` in `RDSPD_ProbaCSP_Conso.csv` must be spelled
   exactly the same in `RDSPD_PartialScore_PD02R3120.csv`. Reading is encoding-robust, but a
   silent mojibake would break the scoring join, not raise an error.
3. **Keep each file's date format.** They differ: `31/03/2005` (day first) in `AggMacro` and
   `Unemployment`, `2024/12/31` (year first) in `InterestRate_Revolving`. Dates are parsed by
   inference, so mixing formats inside one file can silently swap day and month.
4. **Do not reorder rows** in `RDSPD_ScoreParams_*.csv` (three parameters read by position) nor
   move the `Intercept` row away from the top of `RDSPD_PartialScore_*.csv`.
5. **Adding a modality multiplies the grid.** The score grid is the full Cartesian product of
   the modalities: the PD grid is 5 × 5 × 2 × 5 × 3 × 5 = 3 750 rows today. A new modality on a
   risk driver must also be produced by the code that bins that driver (`scores.py`,
   functions `rd_*`), otherwise the rows will not join.
6. **Renaming a modality is a two-file change**: the distribution file *and* the scorecard.

### After editing a scorecard

`tests/test_scores_deterministic.py` compares the generated grids with `reference/ScoreGrid_PD.csv`
and `reference/ScoreGrid_RVV.csv` **bit-for-bit**, so it will fail by design once you change a
scorecard. Either restrict the suite (`pytest -k "not deterministic"`) or refresh the reference
files from `outputs/<version>/ScoreGrid_*.csv` once you have validated the new grids.

Changing anything here changes the simulated portfolio — this is expected. Reproducibility is
guaranteed for a *given* set of parameters and a given `--seed`.
