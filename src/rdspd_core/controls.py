"""Controles de validation du modele.

Reproduit les cellules « Control » du notebook (matrice de covariance, distribution des
variables, distributions conditionnelles, mapping drivers/score, intervalles de confiance
sur P0-P2 / PDt / PD12). Chaque controle est affiche ET enregistre via le :class:`Reporter`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .io_utils import read_csv
from .reporting import Reporter

# Variables numeriques controlees (= varnum du notebook, cellule de controle covariance).
NUM_VARS = ["age", "charges", "maturite", "montant_credit", "nombre_enfants", "revenu"]


def _fmt_pct(df: pd.DataFrame, cols, prec: int = 0) -> pd.DataFrame:
    """Formate des colonnes en pourcentage (remplace ``applymap`` du notebook)."""
    for c in cols:
        df[c] = df[c].map(lambda x: f"{x:.{prec}%}")
    return df


# ---------------------------------------------------------------------------
# Controles de l'octroi
# ---------------------------------------------------------------------------
def control_covariance(data_final: pd.DataFrame, cov: pd.DataFrame, reporter: Reporter) -> None:
    """Ecart entre la covariance simulee et la covariance cible."""
    cov_obs = data_final[NUM_VARS].cov().round(2)
    cov_err = abs(cov - cov_obs)
    reporter.show_df(cov_err, "ctrl_covariance_error", title="Controle - Ecart matrice de covariance :")


def control_describe(df: pd.DataFrame, reporter: Reporter) -> None:
    """Statistiques principales des variables apres application des bornes."""
    reporter.show_df(
        df.describe().astype(int), "ctrl_variable_describe", title="Controle - Distribution des variables :"
    )


def control_conditional_distribution(df: pd.DataFrame, tab_names: dict, reporter: Reporter) -> None:
    """Compare frequences empiriques et theoriques des variables categorielles (IC 95%)."""
    reporter.log("Controle - Distribution conditionnelle (frequence empirique dans l'IC 95%) :")
    for var, tab_path in tab_names.items():
        freq_theo = read_csv(tab_path)
        freq_theo_long = freq_theo.melt(id_vars=["Age_Class"], var_name=var, value_name="p_theo")

        freq_emp = df.groupby(["Age_Class", var]).size().reset_index(name="n")
        total_by_age = df.groupby("Age_Class").size().reset_index(name="total_n")
        freq_emp = freq_emp.merge(total_by_age, on="Age_Class")
        freq_emp["p_emp"] = freq_emp["n"] / freq_emp["total_n"]

        df_final = freq_emp.merge(freq_theo_long, on=["Age_Class", var])

        z = norm.ppf(0.975)
        df_final["IC_lower"] = df_final["p_theo"] - z * np.sqrt(
            (df_final["p_theo"] * (1 - df_final["p_theo"])) / df_final["total_n"]
        )
        df_final["IC_upper"] = df_final["p_theo"] + z * np.sqrt(
            (df_final["p_theo"] * (1 - df_final["p_theo"])) / df_final["total_n"]
        )
        df_final["inside_IC"] = (df_final["p_emp"] >= df_final["IC_lower"]) & (df_final["p_emp"] <= df_final["IC_upper"])

        test_all = bool(df_final.inside_IC.all())
        reporter.log(f"   {var}: All inside - {test_all}")
        if not test_all:
            cols = ["p_emp", "p_theo", "IC_lower", "IC_upper"]
            df_final = _fmt_pct(df_final.copy(), cols, prec=0)
            reporter.show_df(df_final, f"ctrl_condist_{var}_failures")


# ---------------------------------------------------------------------------
# Controles post-simulation
# ---------------------------------------------------------------------------
def control_score_mapping(data: pd.DataFrame, grid: pd.DataFrame, reporter: Reporter) -> None:
    """Verifie que le score simule correspond bien au score attendu de la grille."""
    varlist = ["rd_Age_AncEmp", "rd_DebtRatioW", "situation_habitat", "rd_UsingRate", "rd_CSP", "rd_BankingSen"]
    control = data.query("Defaut==0")[varlist + ["ScorePD"]].drop_duplicates().sort_values("ScorePD")
    control = control.rename(columns={"ScorePD": "ScorePDSimule"})
    control = pd.merge(control, grid[varlist + ["ScorePD"]])
    control["Test"] = control["ScorePDSimule"] == control["ScorePD"]
    control_all = bool(control.Test.all())
    reporter.log("Controle - Mapping drivers / score PD : All inside - " + str(control_all))
    if not control_all:
        reporter.show_df(control, "ctrl_score_mapping_failures")


def prepare_control_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Construit le cadre de controle des probabilites (D0/N0, D1/N1, D2/N2, Dt/Nt)."""
    control = data.query("Master")[
        [
            "Contract_ID", "Creation_dt", "Date", "Month", "ResidualMaturity", "Outstanding", "ScorePD",
            "NbFac", "Retard", "Retard_cumule", "Defaut", "Default12m", "P0", "P1", "P2",
        ]
    ].copy()

    # PDt effectif (avec ajustement potentiel du P0 par rho)
    control = control.assign(PDt=control.P0 * control.P1 * control.P2)

    # P0 : exposition initiale ou retard cumule precedent nul
    control["D0"] = (
        (control.Date == control.Creation_dt) | ((control.Retard_cumule.shift(1) == 0) & (control.NbFac > 0))
    ).astype(int)
    control["N0"] = control.D0 * control.Retard

    # P1
    control["D1"] = (
        (control.Contract_ID == control.Contract_ID.shift(1)) & (control.Retard_cumule.shift(1) == 1)
    ).astype(int)
    control["N1"] = control.D1 * control.Retard

    # P2
    control["D2"] = (
        (control.Contract_ID == control.Contract_ID.shift(2)) & (control.Retard_cumule.shift(1) == 2)
    ).astype(int)
    control["N2"] = control.D2 * control.Retard

    # PDt (V1.3 : on exclut les 2 premieres observations lors du test du PDt)
    control["Dt"] = (
        (control.D0 + control.D1 + control.D2 > 0) & (control.Contract_ID == control.Contract_ID.shift(2))
    ).astype(int)
    control["Nt"] = control.Dt * control.Defaut

    return control


def _var_est(control: pd.DataFrame, perim: str, proba: str) -> pd.DataFrame:
    """Variance estimee de la probabilite moyenne par grade de score."""
    df = control[control[perim] == 1][["ScorePD", proba]].rename(columns={proba: "p"})
    df = df.assign(prod_est=df.p * (1 - df.p))
    df = df.groupby("ScorePD")["prod_est"].agg(sum_prod_est="sum", count_prod_est="count")
    df = df.assign(var_est=df.sum_prod_est / (df.count_prod_est**2))
    df = df.rename(columns={"var_est": proba})
    return df[proba].reset_index()


def control_p0p1p2_pdt(control: pd.DataFrame, grid: pd.DataFrame, reporter: Reporter) -> None:
    """Controle des probabilites P0, P1, P2 et PDt (IC 95% avec ajustement V1.3)."""
    reporter.section("Controle - P0-P2, PDt")

    # Distribution theorique
    freq_theo = grid[["ScorePD", "P0", "P1", "P2", "PDt"]].melt(id_vars=["ScorePD"], var_name="Type", value_name="p_theo")
    freq_theo = freq_theo.drop_duplicates()

    # Probabilites moyennes des exposes (V1.3 : moyenne par grade, tenant compte de rho)
    p0_avg = control[control["D0"] == 1].groupby("ScorePD")["P0"].mean().reset_index()
    p1_avg = control[control["D1"] == 1].groupby("ScorePD")["P1"].mean().reset_index()
    p2_avg = control[control["D2"] == 1].groupby("ScorePD")["P2"].mean().reset_index()
    pdt_avg = control[control["D0"] + control["D1"] + control["D2"] >= 1].groupby("ScorePD")["PDt"].mean().reset_index()
    agg = p0_avg.merge(p1_avg, on="ScorePD", how="outer").merge(p2_avg, on="ScorePD", how="outer").merge(
        pdt_avg, on="ScorePD", how="outer"
    )
    freq_emp = agg.melt(id_vars="ScorePD", var_name="Type", value_name="p_eff")
    df_final = freq_theo.merge(freq_emp, on=["ScorePD", "Type"])

    # Variance estimee (V1.3)
    p0_var = _var_est(control, "D0", "P0")
    p1_var = _var_est(control, "D1", "P1")
    p2_var = _var_est(control, "D2", "P2")
    pdt_var = _var_est(control, "Dt", "PDt")
    agg = p0_var.merge(p1_var, on="ScorePD", how="outer").merge(p2_var, on="ScorePD", how="outer").merge(
        pdt_var, on="ScorePD", how="outer"
    )
    freq_emp = agg.melt(id_vars="ScorePD", var_name="Type", value_name="p_eff_var")
    df_final = df_final.merge(freq_emp, on=["ScorePD", "Type"])

    # Frequences empiriques - denominateur
    agg = control.groupby("ScorePD")[["D0", "D1", "D2", "Dt"]].sum().reset_index()
    agg = agg.rename(columns={"D0": "P0", "D1": "P1", "D2": "P2", "Dt": "PDt"})
    freq_emp = agg.melt(id_vars="ScorePD", var_name="Type", value_name="Den")
    df_final = df_final.merge(freq_emp, on=["ScorePD", "Type"])

    # Frequences empiriques - numerateur
    agg = control.groupby("ScorePD")[["N0", "N1", "N2", "Nt"]].sum().reset_index()
    agg = agg.rename(columns={"N0": "P0", "N1": "P1", "N2": "P2", "Nt": "PDt"})
    freq_emp = agg.melt(id_vars="ScorePD", var_name="Type", value_name="Num")
    df_final = df_final.merge(freq_emp, on=["ScorePD", "Type"])

    # Ratio empirique
    df_final["p_emp"] = df_final["Num"] / df_final["Den"]

    # Intervalle de confiance (V1.3 : moyenne et variance empiriques)
    z = norm.ppf(0.975)
    df_final["IC_lower"] = df_final["p_eff"] - z * np.sqrt(df_final["p_eff_var"])
    df_final["IC_upper"] = df_final["p_eff"] + z * np.sqrt(df_final["p_eff_var"])
    df_final["inside_IC"] = (df_final["p_emp"] >= df_final["IC_lower"]) & (df_final["p_emp"] <= df_final["IC_upper"])

    reporter.log("Nb of unique score grades:  " + str(df_final.ScorePD.nunique()))
    for group in sorted(df_final.Type.unique()):
        df_group = df_final.query("Type==@group").sort_values("ScorePD")
        test_all = bool(df_group.inside_IC.all())
        reporter.log(f"{group}: All inside - {test_all}")
        if not test_all:
            reporter.log("    > Failure:  " + str(len(df_group[df_group.inside_IC == False])))  # noqa: E712
            cols = ["p_theo", "p_eff", "p_emp", "IC_lower", "IC_upper"]
            df_show = _fmt_pct(df_group[df_group.inside_IC == False].copy(), cols, prec=1)  # noqa: E712
            reporter.show_df(df_show, f"ctrl_pdt_{group}_failures")


def _calculate_custom_pd12(group: pd.DataFrame) -> pd.DataFrame:
    """PD12 = 1 - produit(1 - PDt) sur la fenetre glissante de 12 mois."""
    group = group.sort_values("Date")
    pdts = group["PDt"].values
    n = len(pdts)
    pd12_results = np.zeros(n)
    for i in range(n):
        window_end = min(i + 12, n)
        window_vals = [pdts[j] for j in range(i, window_end)]
        pd12_results[i] = 1 - np.prod([1 - v for v in window_vals])
    group["PD12"] = pd12_results
    return group


def control_pd12(control: pd.DataFrame, grid: pd.DataFrame, reporter: Reporter) -> None:
    """Controle de la PD a 12 mois (IC 95%, fenetre glissante - V1.3)."""
    reporter.section("Controle - PD12m")

    freq_theo = grid[["ScorePD", "PD12"]].drop_duplicates()

    control = control.sort_values(["Contract_ID", "Date"])
    control = control.groupby("Contract_ID", group_keys=False).apply(_calculate_custom_pd12)

    control["D12"] = (
        (control.Defaut == 0) & (control.Outstanding > 0) & (control.Month == 12) & (control.ResidualMaturity >= 12)
    ).astype(int)
    control["N12"] = control.D12 * control.Default12m

    agg = control[control["D12"] == 1].groupby("ScorePD")["PD12"].mean().reset_index()
    freq_emp = agg.melt(id_vars="ScorePD", var_name="Type", value_name="p_eff")
    df_final = freq_theo.merge(freq_emp, on=["ScorePD"])

    agg = _var_est(control, "D12", "PD12")
    freq_emp = agg.melt(id_vars="ScorePD", var_name="Type", value_name="p_eff_var")
    df_final = df_final.merge(freq_emp, on=["ScorePD", "Type"])

    freq_emp = control.groupby("ScorePD")[["D12", "N12"]].sum().reset_index()
    freq_emp["p_emp"] = freq_emp["N12"] / freq_emp["D12"]
    df_final = df_final.merge(freq_emp, on="ScorePD")

    z = norm.ppf(0.975)
    df_final["IC_lower"] = df_final["p_eff"] - z * np.sqrt(df_final["p_eff_var"])
    df_final["IC_upper"] = df_final["p_eff"] + z * np.sqrt(df_final["p_eff_var"])
    df_final["inside_IC"] = (df_final["p_emp"] >= df_final["IC_lower"]) & (df_final["p_emp"] <= df_final["IC_upper"])

    cols = ["p_eff", "p_emp", "PD12", "IC_lower", "IC_upper"]
    df_show = _fmt_pct(df_final.copy(), cols, prec=1)
    reporter.log("Nb of unique score grades:  " + str(df_final.ScorePD.nunique()))
    reporter.log("    > Nb of failure:  " + str(len(df_final[df_final.inside_IC == False])))  # noqa: E712
    reporter.show_df(df_show[df_show.inside_IC == False], "ctrl_pd12_failures")  # noqa: E712
