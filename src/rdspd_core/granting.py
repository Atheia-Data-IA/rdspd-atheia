"""Octroi des contrats : generation du portefeuille initial de credits revolving.

Reproduit les cellules « Contract granting » du notebook : generation de donnees
multivariees non normales (generateur IG), affectation des cohortes/annees, de-standardisation,
variables categorielles conditionnelles a l'age, anciennete d'emploi continue, puis
finalisation et export de ``00_Granting.csv``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import controls
from .config import SimulationConfig
from .io_utils import read_csv
from .randnonnormal import IG
from .reporting import Reporter

# Tranches d'age (memes valeurs que le notebook).
AGE_BINS = [0, 24, 29, 39, 49, 59, 69, 79, float("inf")]
AGE_LABELS = ["<25", "25-29", "30-39", "40-49", "50-59", "60-69", "70-79", ">=80"]

# Variables numeriques generees par le generateur IG.
NUM_VARS = ["age", "charges", "maturite", "montant_credit", "nombre_enfants", "revenu"]


# ---------------------------------------------------------------------------
# Lecture et preparation des parametres
# ---------------------------------------------------------------------------
def get_covariance_matrix(path) -> pd.DataFrame:
    """Lit la matrice de covariance (premiere colonne = index)."""
    cov = read_csv(path)
    cov = cov.set_index(cov.columns[0])
    return cov


def process_quantile_data(path, reference_year: int = 2021) -> pd.DataFrame:
    """Derive skewness, kurtosis, ecart-type et moyenne annuels a partir des quantiles."""
    yearly_quantiles = read_csv(path)
    input_data = yearly_quantiles.copy()
    quantile_cols = ["Q25", "Q75", "Q90", "Q10"]
    for col in quantile_cols:
        input_data[col] = pd.to_numeric(input_data[col], errors="coerce")

    # Initialisees en flottant : pandas >= 2.0 interdit l'affectation partielle d'un
    # flottant dans une colonne entiere (le notebook s'appuyait sur l'upcast silencieux).
    input_data["skew"] = 0.0
    input_data["kurt"] = 0.0
    input_data["sd"] = 0.0
    input_data["mean"] = input_data["Q50"]

    data_ref = input_data[input_data["Year"] == reference_year]
    data_ref["skew"] = (data_ref["Q75"] + data_ref["Q25"] - 2 * data_ref["Q50"]) / (data_ref["Q75"] - data_ref["Q25"])
    data_ref["kurt"] = (data_ref["Q75"] - data_ref["Q25"]) / (data_ref["Q90"] - data_ref["Q10"])
    input_data["sd"] = np.where(
        input_data["Q25"].notna(),
        (input_data["Q75"] - input_data["Q25"]) / 1.349,
        (input_data["Q90"] - input_data["Q10"]) / (2 * 1.28),
    )

    skew_kurt_mapping = data_ref.set_index("variable")[["skew", "kurt"]]
    for var in skew_kurt_mapping.index:
        input_data.loc[input_data["variable"] == var, "skew"] = skew_kurt_mapping.loc[var, "skew"]
        input_data.loc[input_data["variable"] == var, "kurt"] = skew_kurt_mapping.loc[var, "kurt"]

    input_data = input_data[["variable", "Year", "skew", "kurt", "sd", "mean"]]
    input_data["sd"] = np.where(input_data["variable"] == "montant_credit", 1000, input_data["sd"])
    input_data["mean"] = np.where(input_data["variable"] == "montant_credit", 3000, input_data["mean"])
    input_data = input_data.fillna(0)

    return input_data


def extract_skewness_kurtosis(input_data: pd.DataFrame, cov: pd.DataFrame, reference_year: int = 2021):
    """Extrait skewness et kurtosis de l'annee de reference pour chaque variable de la covariance."""
    columns = cov.columns
    skewness, kurtosis = [], []
    for col in columns:
        data = input_data.loc[(input_data["Year"] == reference_year) & (input_data["variable"] == col)]
        if not data.empty:
            skewness.append(data["skew"].iloc[0])
            kurtosis.append(data["kurt"].iloc[0])
        else:
            skewness.append(None)
            kurtosis.append(None)
    return skewness, kurtosis


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------
def assign_random_years(data_final: pd.DataFrame, year_weights: pd.DataFrame) -> pd.DataFrame:
    """Affecte une annee d'octroi a chaque contrat selon les poids de cohorte."""
    year_weights = year_weights.copy()
    year_weights["Count"] = year_weights["Weight"] * len(data_final)
    year_weights = year_weights.astype(int)
    years_list = []
    for _, row in year_weights.iterrows():
        years_list.extend([row["Year"]] * row["Count"])
    data_final["Year"] = np.random.choice(years_list, size=len(data_final), replace=False)
    return data_final


def destandarization(df: pd.DataFrame, variables, input_data: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """De-standardise les variables via moyenne/ecart-type annuels (cas particulier des charges).

    On conserve la boucle ligne a ligne et la graine interne du notebook pour preserver
    l'ordre exact des tirages aleatoires.
    """
    np.random.seed(random_seed)

    mean_lookup = input_data.pivot(index="Year", columns="variable", values="mean").to_dict()
    sd_lookup = input_data.pivot(index="Year", columns="variable", values="sd").to_dict()

    alpha_lookup = {year: np.random.uniform(0.2, 0.35) for year in df["Year"].unique()}

    for index, row in df.iterrows():
        year = row["Year"]
        for var in variables:
            if var != "charges":
                mean = mean_lookup[var].get(year, np.nan)
                sd = sd_lookup[var].get(year, np.nan)
            else:
                alpha = alpha_lookup[year]
                mean = alpha * mean_lookup["revenu"].get(year, np.nan)
                sd = alpha * sd_lookup["revenu"].get(year, np.nan)
            if not pd.isna(mean) and not pd.isna(sd):
                df.at[index, var] = (row[var] * sd) + mean

    return df


def simul_bankingsen(df: pd.DataFrame) -> pd.DataFrame:
    """Simule l'anciennete bancaire (loi de Poisson dependant de l'age)."""
    conditions = [
        df.age < 25,
        (df.age >= 25) & (df.age < 30),
        (df.age >= 30) & (df.age < 35),
        (df.age >= 35) & (df.age < 40),
        (df.age >= 40) & (df.age < 50),
        (df.age >= 50) & (df.age < 60),
        (df.age >= 60) & (df.age < 70),
    ]
    labels = [
        np.random.poisson(lam=2),
        np.random.poisson(lam=7),
        np.random.poisson(lam=10),
        np.random.poisson(lam=12),
        np.random.poisson(lam=15),
        np.random.poisson(lam=17),
        np.random.poisson(lam=20),
    ]
    default = np.random.poisson(lam=22)
    df["banking_seniority"] = np.select(conditions, labels, default=default)
    return df


def enforce_limits(df: pd.DataFrame, limits: dict) -> pd.DataFrame:
    """Applique les bornes min/max et les arrondis metier sur les variables."""
    for column, (min_limit, max_limit) in limits.items():
        if column in df.columns:
            df[column] = df[column].clip(lower=min_limit, upper=max_limit)
        else:
            print(f"Column '{column}' not found in DataFrame.")

    # Si revenu == 500 alors charges == 100
    df["charges"] = np.where(df["revenu"] == 500, 100, df["charges"])
    # Montant de credit : arrondi a la centaine, en centimes
    df["montant_credit"] = (df["montant_credit"] / 50).astype(int) * 50 * 100
    # Maturite : arrondie a 12 mois
    df["maturite"] = (df["maturite"] / 12).astype(int) * 12
    # Age, nb enfants : arrondis
    df[["age", "nombre_enfants"]] = df[["age", "nombre_enfants"]].astype(int)

    return df


def categorize_age(df: pd.DataFrame, age_column: str, bins, labels) -> pd.DataFrame:
    """Cree la colonne ``Age_Class`` par decoupage de l'age."""
    if len(bins) - 1 != len(labels):
        raise ValueError("The length of 'labels' should be one less than the length of 'bins'.")
    df["Age_Class"] = pd.cut(df[age_column], bins=bins, labels=labels, right=False)
    return df


def cat_variables(df: pd.DataFrame, values_age, tab_names: dict) -> pd.DataFrame:
    """Affecte les variables categorielles (CSP, habitat, situation, emploi) selon l'age.

    ``values_age`` est passe dans un ordre deterministe (cf. :func:`run_granting`) pour
    garantir la reproductibilite des tirages a graine fixee.
    """
    for category, tab_name in tab_names.items():
        proba_data = read_csv(tab_name)
        proba_data = proba_data.set_index(proba_data.columns[0])
        df[category] = ""  # colonne objet (le notebook initialisait a 0 puis upcastait)
        dataframe_list = []

        for value in values_age:
            df_1 = df.loc[df["Age_Class"] == value].reset_index(drop=True)
            n = len(df_1)
            categories = list(proba_data.columns)
            p = list(proba_data.loc[value])
            # Un tirage par ligne, dans le meme ordre que le notebook (sequence RNG identique),
            # mais affecte en une seule fois pour la performance.
            draws = [np.random.choice(categories, 1, replace=True, p=p)[0] for _ in range(n)]
            df_1[category] = pd.Series(draws, index=df_1.index, dtype=object)
            dataframe_list.append(df_1)

        data_1 = pd.concat(dataframe_list)
        df = data_1.sort_values(by="id")

    return df


def generate_random_employment_years(age_cat, anc_emp_cat):
    """Genere une anciennete d'emploi continue coherente avec l'age et sa categorie."""
    max_year = {
        "<25": 7, "25-29": 10, "30-39": 20, "40-49": 30,
        "50-59": 40, "60-69": 45, "70-79": 50, ">=80": 55,
    }[age_cat]

    if anc_emp_cat == "<1":
        return 0
    elif anc_emp_cat == "1-5":
        if max_year < 1:
            return max_year
        return np.random.randint(1, min(5, max_year) + 1)
    elif anc_emp_cat == "5-10":
        if max_year < 5:
            return max_year
        return np.random.randint(5, min(10, max_year) + 1)
    elif anc_emp_cat == ">= 10":
        if max_year < 10:
            return max_year
        return np.random.randint(10, max_year + 1)


def random_date(year: int) -> pd.Timestamp:
    """Tire une date aleatoire (dernier jour d'un mois aleatoire) dans l'annee donnee."""
    random_month = np.random.randint(1, 13)
    days_in_month = (datetime(year, random_month, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    return pd.to_datetime(days_in_month)


# ---------------------------------------------------------------------------
# Orchestration de l'octroi
# ---------------------------------------------------------------------------
def run_granting(cfg: SimulationConfig, reporter: Reporter) -> pd.DataFrame:
    """Genere le portefeuille initial, emet les controles et exporte ``00_Granting.csv``."""
    reporter.section("Octroi des contrats (granting)")

    # Parametres de generation
    cov = get_covariance_matrix(cfg.input_path("RDSPD_CovarMatrix_Conso.csv"))
    input_data = process_quantile_data(cfg.input_path("RDSPD_Quantile_Conso.csv"), reference_year=cfg.reference_year)
    skewness, kurtosis = extract_skewness_kurtosis(input_data, cov, reference_year=cfg.reference_year)

    # Generation des donnees non normales
    reporter.log(f"Generation de {cfg.n_contracts} contrats (generateur IG)...")
    data_final = IG(cfg.n_contracts, cov, skewness, kurtosis, cfg.reps, cfg.type_a)[0]
    data_final = data_final.astype(float).reset_index(drop=True)

    # Controle - matrice de covariance
    controls.control_covariance(data_final, cov, reporter)

    # Affectation des annees d'octroi (cohortes)
    year_weights = read_csv(cfg.input_path("RDSPD_Cohort_Revolving.csv"))
    year_weights = year_weights.query("(Year > 2005) & (Year < 2024)")
    year_weights["Weight"] = year_weights["Weight"] / year_weights["Weight"].sum()
    year_weights["Weight"] = round(100 * year_weights["Weight"], 2)
    data_final = assign_random_years(data_final, year_weights)

    # De-standardisation + anciennete bancaire
    df = data_final.copy()
    variables = ["age", "nombre_enfants", "revenu", "maturite", "charges", "montant_credit"]
    df = destandarization(df, variables, input_data, random_seed=cfg.destandarization_seed)
    df = simul_bankingsen(df)

    # Bornes metier
    limits = {
        "age": (18, 85),
        "nombre_enfants": (0, float("inf")),
        "revenu": (500, float("inf")),
        "charges": (100, float("inf")),
        "montant_credit": (150, 10000),
        "maturite": (12, 90),
    }
    df = enforce_limits(df, limits)

    # Controle - distribution des variables
    controls.control_describe(df, reporter)

    # Index contrat + tranches d'age
    df = df.reset_index().rename(columns={"index": "id"})
    df = categorize_age(df, "age", AGE_BINS, AGE_LABELS)

    # Variables categorielles conditionnelles a l'age
    tab_names = {
        "situation_habitat": cfg.input_path("RDSPD_ProbaHousing_Conso.csv"),
        "situation_matrimoniale": cfg.input_path("RDSPD_ProbaFamily_Conso.csv"),
        "CSP": cfg.input_path("RDSPD_ProbaCSP_Conso.csv"),
        "anc_emploi": cfg.input_path("RDSPD_ProbaEmploy_Conso.csv"),
    }
    present = set(df["Age_Class"])
    values_age = [v for v in AGE_LABELS if v in present]  # ordre deterministe
    df = cat_variables(df, values_age, tab_names)
    reporter.show_df(
        df[["id", "situation_habitat", "situation_matrimoniale", "CSP", "anc_emploi"]].head(5),
        "granting_categorical_head",
        title="Apercu des variables categorielles :",
    )

    # Anciennete d'emploi continue
    df["anc_emp_continuous"] = df.apply(
        lambda row: generate_random_employment_years(row["Age_Class"], row["anc_emploi"]), axis=1
    )

    # Controle - distribution conditionnelle
    controls.control_conditional_distribution(df, tab_names, reporter)

    # Finalisation : selection et renommage des colonnes
    data = df[
        [
            "id", "Year", "age", "Age_Class", "nombre_enfants", "situation_matrimoniale",
            "situation_habitat", "revenu", "charges", "CSP", "anc_emploi", "anc_emp_continuous",
            "montant_credit", "maturite", "banking_seniority",
        ]
    ].copy()
    data.columns = [
        "Contract_ID", "Annee", "Age", "Age_Categorie", "Nombre_enfant", "Situation_matrimoniale",
        "situation_habitat", "Revenu", "Charges", "CSP", "anciennete_emploi", "anc_emp_continuous",
        "plafond_credit", "Maturite", "banking_seniority",
    ]

    # Dates aleatoires
    data["Date"] = data["Annee"].apply(random_date)
    data["Month"] = data["Date"].dt.month
    data["Birth_dt"] = (data["Annee"] - data["Age"]).apply(random_date)
    data["Employment_dt"] = (data["Annee"] - data["anc_emp_continuous"]).apply(random_date)
    data["Banking_dt"] = (data["Annee"] - data["banking_seniority"]).apply(random_date)

    # Export
    out = cfg.output_path("00_Granting.csv")
    data.to_csv(out, index=False, encoding="utf-8")
    reporter.log(f"Octroi exporte : {out}  ({len(data)} contrats)")

    return data
