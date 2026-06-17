"""Construction et application des grilles de score (revolving RVV et defaut PD).

Reproduit les cellules « Scores » du notebook :

- :func:`build_rvv_grid` et :func:`build_pd_grid` construisent les grilles de score par
  produit cartesien des modalites, calculent le score, les cotes (odds) et les probabilites
  theoriques, puis **exportent** les grilles en CSV.
- Les fonctions ``*score_rd`` calculent/binnent les drivers de risque sur un snapshot.
- Les fonctions ``*score_apply`` rattachent la grille au snapshot et tirent les evenements
  (drawing revolving, impayes / defaut conditionnels a P0/P1/P2 et a l'autocorrelation rho).

IMPORTANT (fidelite) : les grilles construites en memoire ont des colonnes de modalites
typees en chaine (ex. ``dec_flag`` = '0'/'1'). Lors de l'application, le snapshot calcule
``dec_flag`` en entier. Le notebook fonctionne parce qu'il **relit les grilles depuis le CSV**
(``init_grid``), ce qui re-infere ``dec_flag`` en entier. On preserve donc ce round-trip :
``build_*_grid`` exporte les CSV, et la simulation relit les grilles via ``snapshots.init_grid``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm

from .config import SimulationConfig
from .io_utils import read_csv
from .macro import increment_aggmacro

# Cles et colonnes de la grille PD utilisees a l'application.
PD_KEYLIST = ["rd_Age_AncEmp", "rd_DebtRatioW", "situation_habitat", "rd_UsingRate", "rd_CSP", "rd_BankingSen"]
PD_VARLIST = ["ScorePD", "PD12", "PDt", "P0", "P1", "P2"]

# Cles et colonnes de la grille RVV utilisees a l'application.
RVV_KEYLIST = ["dec_flag", "dispo_ratio_bin", "draw_senior_bin"]
RVV_VARLIST = ["ScoreRVV", "RVV12", "RVVt"]


# ---------------------------------------------------------------------------
# Construction des grilles
# ---------------------------------------------------------------------------
def build_rvv_grid(cfg: SimulationConfig) -> pd.DataFrame:
    """Construit et exporte la grille de score revolving (RVV)."""
    sc = cfg.scorecard_rvv
    score_partial = read_csv(cfg.input_path(f"RDSPD_PartialScore_{sc}.csv"))
    score_params = read_csv(cfg.input_path(f"RDSPD_ScoreParams_{sc}.csv"))

    varlist = RVV_KEYLIST
    mods = [score_partial[score_partial["Var"] == v]["Mod"].unique() for v in varlist]
    grid = pd.DataFrame(itertools.product(*mods), columns=varlist)

    intercept = score_partial["PartialScore"].iloc[0]
    grid["intercept"] = intercept
    for var in varlist:
        right = score_partial.query("Var==@var")[["Mod", "PartialScore"]].rename(
            columns={"Mod": var, "PartialScore": "w_" + var}
        )
        grid = pd.merge(grid, right, on=var, how="left")

    score_ampli = score_params["Value"].iloc[0]
    odds_ratio = score_params["Value"].iloc[1]
    score_iso = score_params["Value"].iloc[2]

    grid["ScoreRVV"] = round(grid.intercept + grid.w_dec_flag + grid.w_dispo_ratio_bin + grid.w_draw_senior_bin, 2)
    grid["Odds"] = odds_ratio ** ((grid.ScoreRVV - score_iso) / score_ampli)
    grid["RVV12"] = round(grid.Odds / (1 + grid.Odds), 3)
    grid["RVVt"] = 1 - (1 - grid["RVV12"]) ** (1 / 12)

    grid.to_csv(cfg.output_path("ScoreGrid_RVV.csv"), index=False, encoding="utf-8")
    return grid


def _objective_with_constraints(p, PDt):
    """Fonction objectif penalisee pour la decomposition PDt = P0 * P1 * P2."""
    P0, P1, P2 = p
    penalty = 0
    if P1 <= 2 * P0 or P1 <= P2:
        penalty += 1000  # Penalite forte si contrainte violee
    if P0 < 0.005:
        penalty += 1000
    return abs(PDt - P0 * P1 * P2) + penalty


def _optimize_probabilities(pdt: float):
    """Optimise (P0, P1, P2) pour une valeur de PDt donnee."""
    initial_guess = [0.1, 0.5, 0.7]
    bounds = [(0, 1), (0, 1), (0, 1)]
    result = minimize(_objective_with_constraints, initial_guess, bounds=bounds, args=(pdt,))
    return result.x


def build_pd_grid(cfg: SimulationConfig, reporter=None) -> pd.DataFrame:
    """Construit et exporte la grille de score de defaut (PD), avec optimisation P0/P1/P2."""
    sc = cfg.scorecard_pd
    score_partial = read_csv(cfg.input_path(f"RDSPD_PartialScore_{sc}.csv"))
    score_params = read_csv(cfg.input_path(f"RDSPD_ScoreParams_{sc}.csv"))

    varlist = PD_KEYLIST
    mods = [score_partial[score_partial["Var"] == v]["Mod"].unique() for v in varlist]
    grid = pd.DataFrame(itertools.product(*mods), columns=varlist)
    if reporter is not None:
        reporter.log(f"Card length:  {len(grid)}")

    intercept = score_partial["PartialScore"].iloc[0]
    grid["intercept"] = intercept
    for var in varlist:
        right = score_partial.query("Var==@var")[["Mod", "PartialScore"]].rename(
            columns={"Mod": var, "PartialScore": "w_" + var}
        )
        grid = pd.merge(grid, right, on=var, how="left")

    score_ampli = score_params["Value"].iloc[0]
    odds_ratio = score_params["Value"].iloc[1]
    score_iso = score_params["Value"].iloc[2]

    grid["ScorePD"] = round(
        grid.intercept
        + grid.w_rd_Age_AncEmp
        + grid.w_rd_DebtRatioW
        + grid.w_situation_habitat
        + grid.w_rd_UsingRate
        + grid.w_rd_CSP
        + grid.w_rd_BankingSen,
        2,
    )
    grid["Odds"] = odds_ratio ** ((grid.ScorePD - score_iso) / score_ampli)
    grid["PD12"] = round(grid.Odds / (1 + grid.Odds), 3)
    grid["PDt"] = 1 - (1 - grid["PD12"]) ** (1 / 12)

    # Optimisation de (P0, P1, P2) pour chaque valeur unique de PDt.
    # (Le notebook utilisait groupby('PDt').apply ; on itere explicitement pour rester
    #  compatible avec les versions recentes de pandas, le resultat est identique.)
    gridunique = grid[["PDt"]].drop_duplicates().reset_index(drop=True)
    optimized = np.array([_optimize_probabilities(pdt) for pdt in tqdm(gridunique["PDt"], desc="Processing Scorecard")])
    gridunique[["P0", "P1", "P2"]] = optimized
    grid = grid.merge(gridunique, on="PDt", how="left")

    # Controle : P0 * P1 * P2 == PDt
    if reporter is not None:
        control = grid.copy()
        control["test0"] = control["P0"] * control["P1"] * control["P2"]
        control["test1"] = abs(control["test0"] - control["PDt"]) < 10**-5
        reporter.log("Controle grille PD - P0 x P1 x P2 = PDt : tous valides = " + str(bool(control.test1.all())))
        failures = control[~control.test1]
        if len(failures):
            reporter.show_df(failures, "ctrl_pd_grid_p0p1p2_failures", save=True)

    grid.to_csv(cfg.output_path("ScoreGrid_PD.csv"), index=False, encoding="utf-8")
    return grid


# ---------------------------------------------------------------------------
# Application des scores - Revolving (RVV)
# ---------------------------------------------------------------------------
def rvvscore_rd(updt: pd.DataFrame, check: bool = False) -> pd.DataFrame:
    """Calcule et binne les drivers de risque du score revolving."""
    # dec_flag : penalite de fin d'annee (mois de decembre)
    updt["dec_flag"] = np.where(updt.Month == 12, 1, 0)

    # dispo_ratio_bin : ratio de disponibilite
    updt["dispo_ratio"] = (updt.Disposable / updt.plafond_credit * 100).astype(int)
    bins = [0, 20, 50, 80, float("inf")]
    labels = ["< 0.2", "0.2 - 0.5", "0.5 - 0.8", "> 0.8"]
    updt["dispo_ratio_bin"] = pd.cut(updt["dispo_ratio"], bins=bins, labels=labels, right=False)

    # draw_senior_bin : anciennete du tirage (mois ecoules depuis la creation)
    updt["draw_senior"] = (updt.Date.dt.year - updt.Creation_dt.dt.year) * 12 + (
        updt.Date.dt.month - updt.Creation_dt.dt.month
    )
    bins = [0, 12, 24, 36, float("inf")]
    labels = ["< 12", "12 - 24", "24 - 36", "> 36"]
    updt["draw_senior_bin"] = pd.cut(updt["draw_senior"], bins=bins, labels=labels, right=False)

    return updt


def rvvscore_apply(df: pd.DataFrame, grid: pd.DataFrame, keylist=None, varlist=None) -> pd.DataFrame:
    """Rattache la grille RVV et tire l'evenement de tirage (drawing)."""
    keylist = keylist or RVV_KEYLIST
    varlist = varlist or RVV_VARLIST

    df = df.drop(columns=varlist)
    df = pd.merge(df, grid[keylist + varlist], on=keylist, how="left").set_index(df.index)

    # Tirage du drawing
    df["Drawing"] = np.random.binomial(1, df.RVVt, len(df)).astype(int)

    # Override - pas de tirage si disponible < 150 (en centimes : 15000)
    df["Drawing_override"] = np.where(df.Disposable < 15000, 1, 0).astype(int)
    # Override - pas de tirage si retard
    df["Drawing_override"] = np.where(df.Retard == 1, 1, df.Drawing_override).astype(int)
    # Application de l'override
    df["Drawing"] = np.where(df.Drawing_override == 1, 0, df.Drawing).astype(int)

    return df


# ---------------------------------------------------------------------------
# Application des scores - Defaut (PD02R3120)
# ---------------------------------------------------------------------------
def pdscore_rd(snap: pd.DataFrame, check: bool = False) -> pd.DataFrame:
    """Calcule et binne les drivers de risque du score de defaut (scorecard PD02R3120)."""
    # Variables intermediaires
    snap["DisposableIncome"] = snap.Revenu_ajuste - snap.Charges_ajuste
    # Chaque personne supplementaire compte pour 1/2 part
    snap["HouseSize"] = np.where(snap.Situation_matrimoniale == "Single", 1, 1.5) + snap.Nombre_enfant / 2
    snap["DisposableIncomeW"] = snap.DisposableIncome / snap.HouseSize

    # Drivers continus
    snap["DebtRatioW"] = snap.performing_claim / snap.DisposableIncomeW
    snap["UsingRate"] = snap.Outstanding / snap.plafond_credit

    # Age x anciennete d'emploi
    conditions = [
        (snap.Age < 40) & (snap.anc_emp_continuous < 5),
        (snap.Age >= 60) & (snap.anc_emp_continuous < 5),
        (snap.Age < 40) & (snap.anc_emp_continuous >= 5),
        (snap.Age >= 60) & (snap.anc_emp_continuous >= 5),
    ]
    labels = [
        "Age < 40. Emp < 5",
        "Age >= 60. Emp < 5",
        "Age < 40. Emp >= 5",
        "Age >= 60. Emp >=5",
    ]
    snap["rd_Age_AncEmp"] = np.select(conditions, labels, default="40 <= Age < 60")

    # Taux d'endettement (pondere par la composition du foyer)
    bins = [float("-inf"), 0, 0.05, 0.10, 0.15, 0.3, float("inf")]
    labels = ["> .3", "< .05", "< .1", "< .15", "< .3", "> .3"]
    snap["rd_DebtRatioW"] = pd.cut(snap["DebtRatioW"], bins=bins, labels=labels, right=False, ordered=False)

    # Taux d'utilisation
    bins = [0, 0.1, 0.2, 0.3, 0.5, float("inf")]
    labels = ["< 10%", "< 20%", "< 30%", "< 50%", "> 50%"]
    snap["rd_UsingRate"] = pd.cut(snap["UsingRate"], bins=bins, labels=labels, right=False)

    # Categorie socio-professionnelle (CSP) regroupee
    conditions = [
        snap.CSP == "Cadres",
        snap.CSP == "Indépendants",
        snap.CSP == "Employés",
        snap.CSP == "Professions intermédiaires",
        snap.CSP == "Ouvriers",
        snap.CSP == "Agriculteurs",
    ]
    labels = [
        "Cadres. Indépendants",
        "Cadres. Indépendants",
        "Employés. Professions intermédiaires",
        "Employés. Professions intermédiaires",
        "Ouvriers. Agriculteurs",
        "Ouvriers. Agriculteurs",
    ]
    snap["rd_CSP"] = np.select(conditions, labels, default="Error")

    # Anciennete bancaire
    bins = [0, 5, 10, 15, 20, float("inf")]
    labels = ["< 5", "< 10", "< 15", "< 20", "> 20"]
    snap["rd_BankingSen"] = pd.cut(snap["banking_seniority"], bins=bins, labels=labels, right=False)

    return snap


def pdscore_apply(
    df: pd.DataFrame,
    grid: pd.DataFrame,
    keylist=None,
    varlist=None,
    rho: float = 0.0,
    aggregate: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rattache la grille PD et tire les impayes / le defaut.

    Si ``rho > 0``, le P0 est ajuste par l'autocorrelation des actifs (modele a facteur
    unique), via le facteur macro agrege.
    """
    keylist = keylist or PD_KEYLIST
    varlist = varlist or PD_VARLIST

    df = df.drop(columns=varlist)
    df = pd.merge(df, grid[keylist + varlist], on=keylist, how="left").set_index(df.index)

    # Impact de l'autocorrelation sur le P0 conditionnel
    if rho > 0:
        if aggregate is None:
            raise ValueError("aggregate est requis lorsque rho > 0")
        df = increment_aggmacro(df, aggregate)
        df["q"] = norm.ppf(df.P0, loc=0, scale=1)
        df["P0"] = norm.cdf((df.q - np.sqrt(rho) * df.AggMacro) / np.sqrt(1 - rho), loc=0, scale=1)

    # Tirage des impayes courants conditionnels a P0 (si rho==0 alors P0y=P0)
    df["Retard"] = np.where(
        (df.Retard_cumule == 0) & (df.Outstanding > 0), np.random.binomial(n=1, p=df.P0), df.Retard
    )
    # Conditionnels a P1
    df["Retard"] = np.where(df.Retard_cumule == 1, np.random.binomial(n=1, p=df.P1), df.Retard)
    # Conditionnels a P2
    df["Retard"] = np.where(df.Retard_cumule == 2, np.random.binomial(n=1, p=df.P2), df.Retard)

    # Incrementation du compteur d'impayes consecutifs
    df["Retard_cumule"] = np.where(
        df.Retard_cumule < 3,
        df["Retard_cumule"] * df["Retard"] + df["Retard"],
        df.Retard_cumule,
    )

    # Declenchement du defaut (3 impayes consecutifs)
    df["Defaut"] = np.where(df.Retard_cumule == 3, 1, df.Defaut)

    return df
