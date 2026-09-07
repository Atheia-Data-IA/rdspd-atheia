"""Import des facteurs macro-economiques.

Reproduit les cellules « Macro-economic factors import » du notebook : taux de chomage
et durees, indices d'inflation, indices de revenu par CSP, taux d'interet revolving,
et facteur macro-economique agrege. Toutes ces series sont regroupees dans
:class:`MacroData` pour etre passees explicitement aux fonctions de simulation
(plutot que via des variables globales comme dans le notebook).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import SimulationConfig
from .io_utils import read_csv


@dataclass
class MacroData:
    """Conteneur des series macro-economiques utilisees par la simulation."""

    unemp_rates: pd.DataFrame      # colonnes : Date, Mu_t
    duration: list                 # durees de chomage possibles (D du notebook)
    duration_proba: list           # probabilites associees (P du notebook)
    inflation_df: pd.DataFrame     # colonnes : Year, Base
    income_df: pd.DataFrame        # format long : Annee, CSP, Base
    interest_rates: pd.DataFrame   # colonnes : Date, InterestRateAvg
    aggregate: pd.DataFrame        # colonnes : Date, AggMacro, Quarter


def load_macro(cfg: SimulationConfig) -> MacroData:
    """Charge l'ensemble des facteurs macro-economiques depuis ``inputs/``."""
    # Taux de chomage et durees
    unemp_rates = read_csv(cfg.input_path("RDSPD_Unemployment.csv"))
    unemp_rates["Date"] = pd.to_datetime(unemp_rates["Date"])
    unemp_duration = read_csv(cfg.input_path("RDSPD_Unemployment_Duration.csv"))
    duration = list(unemp_duration.Duration)
    duration_proba = list(unemp_duration.Proba)

    # Indices d'inflation (base 2010)
    inflation_df = read_csv(cfg.input_path("RDSPD_Inflation.csv"))

    # Indices de revenu (euros constants, ajustes de l'inflation) -> format long
    income_df = read_csv(cfg.input_path("RDSPD_IncomeIndex.csv"))
    income_df = income_df.melt(id_vars=["Annee"], var_name="CSP", value_name="Base")

    # Taux d'interet revolving
    interest_rates = read_csv(cfg.input_path("RDSPD_InterestRate_Revolving.csv"))
    interest_rates["Date"] = pd.to_datetime(interest_rates["Date"])

    # Facteur macro-economique agrege (filtre sur 2006-2023, comme le notebook)
    aggregate = read_csv(cfg.input_path("RDSPD_AggMacro.csv"))
    aggregate["Quarter"] = pd.to_datetime(aggregate["Date"])
    aggregate = aggregate.query("(Quarter >= '2006-01-31') & (Quarter <= '2023-12-31')")

    return MacroData(
        unemp_rates=unemp_rates,
        duration=duration,
        duration_proba=duration_proba,
        inflation_df=inflation_df,
        income_df=income_df,
        interest_rates=interest_rates,
        aggregate=aggregate,
    )


def increment_aggmacro(df: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    """Rattache a chaque ligne le facteur macro agrege du trimestre courant."""
    if "AggMacro" in df.columns:
        df = df.drop(columns=["AggMacro"])

    df["Quarter"] = df["Date"] + pd.offsets.QuarterEnd(0)
    df = df.merge(aggregate[["Quarter", "AggMacro"]], on="Quarter", how="inner").set_index(df.index)
    del df["Quarter"]

    return df
