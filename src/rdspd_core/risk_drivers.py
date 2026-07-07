"""Mise a jour des facteurs de risque (drivers) a chaque snapshot mensuel.

Reproduit les cellules « Risk drivers update » du notebook : dates, chomage, revenus
et charges ajustes de l'inflation, age, anciennete d'emploi, anciennete bancaire, et
agregation des encours au niveau du contrat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def increment_dates(updt: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
    """Met a jour la date courante (Date / Annee / Month) des lignes a actualiser."""
    updt["Date"] = pd.to_datetime(dt)
    updt["Annee"] = dt.year
    updt["Month"] = dt.month
    return updt


def increment_unemp(
    updt: pd.DataFrame,
    dt: pd.Timestamp,
    unemp_rates: pd.DataFrame,
    duration: list,
    duration_proba: list,
) -> pd.DataFrame:
    """Simule les migrations vers/depuis le chomage et la duree restante."""
    # Taux de migration (rempli trimestriellement)
    Mu_t = 0
    if dt in list(unemp_rates["Date"]):
        Mu_t = unemp_rates[unemp_rates["Date"] == dt]["Mu_t"].iloc[0]

    # Migration vers le chomage
    updt["_NewUnemp"] = 0
    updt.loc[updt["Unemp"] == 0, "_NewUnemp"] = np.random.binomial(
        1, Mu_t, (updt["Unemp"] == 0).sum()
    )
    updt.loc[updt["_NewUnemp"] == 1, "Duration"] = np.random.choice(
        duration, size=(updt["_NewUnemp"] == 1).sum(), replace=True, p=duration_proba
    )

    # Migration vers l'emploi
    updt.loc[(updt["Unemp"] == 1) & (updt["_NewUnemp"] == 0), "Duration"] -= 1
    updt.loc[(updt["Unemp"] == 1) & (updt["Duration"] == 0), ["Unemp", "Employment_dt"]] = [0, dt]

    # Nettoyage
    updt.loc[updt["_NewUnemp"] == 1, "Unemp"] = 1
    del updt["_NewUnemp"]

    return updt


def increment_income_wages(
    updt: pd.DataFrame,
    inflation_df: pd.DataFrame,
    income_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ajuste revenus et charges via les indices de prix et de revenu par CSP.

    Optimisation (neutre) : les 4 merges par appel sont remplaces par des lookups
    (``map`` / ``reindex`` sur des cles uniques). Les indices d'inflation (par annee) et de
    revenu (par (annee, CSP)) etant uniques, ``merge how='left'`` et ``map`` renvoient
    exactement les memes valeurs (NaN pour les cles absentes). Resultat byte-a-byte identique.
    """
    # --- Inflation : dictionnaire annee -> indice (colonnes positionnelles [annee, indice]) ---
    infl_map = dict(zip(inflation_df.iloc[:, 0], inflation_df.iloc[:, 1], strict=True))
    base_index = updt["Creation_yr"].map(infl_map)
    current_index = updt["Annee"].map(infl_map)

    # Charges ajustees
    updt["Charges_ajuste"] = updt["Charges"] * (current_index / base_index)
    updt["Charges_ajuste"] = (updt["Charges_ajuste"] * 100).astype(int)

    # --- Revenu : indice par (annee, CSP) (colonnes positionnelles [annee, CSP, indice]) ---
    inc_ser = income_df.set_index(list(income_df.columns[:2]))[income_df.columns[2]]
    base_income = pd.Series(
        inc_ser.reindex(pd.MultiIndex.from_arrays([updt["Creation_yr"], updt["CSP"]])).to_numpy(),
        index=updt.index,
    )
    current_income = pd.Series(
        inc_ser.reindex(pd.MultiIndex.from_arrays([updt["Annee"], updt["CSP"]])).to_numpy(),
        index=updt.index,
    )

    # Revenu ajuste (les chomeurs percoivent une allocation forfaitaire de 750).
    ratio = (current_index / base_index) * (current_income / base_income)
    updt["Revenu_ajuste"] = np.where(updt["Unemp"] == 1, 750 * ratio, updt["Revenu"] * ratio)
    updt["Revenu_ajuste"] = (updt["Revenu_ajuste"] * 100).astype(int)

    return updt


def increment_age(updt: pd.DataFrame) -> pd.DataFrame:
    """Recalcule l'age et sa categorie."""
    updt["Age"] = ((updt.Date - updt.Birth_dt).dt.days / 365.25).astype(int)
    bins = [0, 24, 29, 39, 49, 59, 69, 79, float("inf")]
    labels = ["<25", "25-29", "30-39", "40-49", "50-59", "60-69", "70-79", ">=80"]
    updt["Age_Categorie"] = pd.cut(updt.Age, bins=bins, labels=labels, right=False)
    return updt


def increment_empsen(updt: pd.DataFrame) -> pd.DataFrame:
    """Recalcule l'anciennete d'emploi (continue et categorisee)."""
    updt["anc_emp_continuous"] = ((updt.Date - updt.Employment_dt).dt.days / 365.25).astype(int)
    bins = [0, 1, 5, 10, float("inf")]
    labels = ["<1", "1-5", "5-10", ">= 10"]
    updt["anciennete_emploi"] = pd.cut(updt.anc_emp_continuous, bins=bins, labels=labels, right=False)
    return updt


def increment_banking_seniority(updt: pd.DataFrame) -> pd.DataFrame:
    """Recalcule l'anciennete bancaire."""
    updt["banking_seniority"] = ((updt.Date - updt.Banking_dt).dt.days / 365.25).astype(int)
    return updt


def increment_aggregates(master: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """Agrege les encours des facilites (sub) au niveau du contrat (master)."""
    aggregations = {
        "Creation_dt": ["count"],
        "montant_credit": ["sum"],
        "performing_claim": ["sum"],
        "Principal": ["sum"],
        "Disposable": ["sum"],
        "Outstanding": ["sum"],
        "ResidualMaturity": ["max"],
        "interest_rate_month": ["max"],
    }

    df = sub.groupby("Contract_ID").agg(aggregations)
    df.columns = [
        "NbFac",
        "montant_credit",
        "performing_claim",
        "Principal",
        "Disposable",
        "Outstanding",
        "ResidualMaturity",
        "interest_rate_month",
    ]

    master[
        [
            "NbFac",
            "montant_credit",
            "performing_claim",
            "Principal",
            "Disposable",
            "Outstanding",
            "ResidualMaturity",
        ]
    ] = 0
    # interest_rate_month est un taux (flottant) : on l'initialise en flottant.
    master["interest_rate_month"] = 0.0

    # Optimisation (neutre) : remplace master.update(df). df provient d'une agregation (aucun NaN)
    # et son index est inclus dans celui de master ; l'affectation colonne par colonne sur
    # l'intersection reproduit exactement update() tout en preservant les dtypes.
    idx = master.index.intersection(df.index)
    for col in df.columns:
        master.loc[idx, col] = df.loc[idx, col]

    return master
