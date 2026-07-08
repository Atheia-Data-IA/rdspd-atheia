"""Orchestration de bout-en-bout de la simulation RDS PD.

Enchaine : octroi -> grilles de score -> initialisation -> boucle de simulation ->
consolidation -> controles -> exports. C'est le point d'entree unique destine a etre
appele aussi bien par la CLI que, plus tard, par une API.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import controls
from .config import SimulationConfig
from .granting import run_granting
from .io_utils import read_csv
from .macro import load_macro
from .reporting import Reporter
from .scores import build_pd_grid, build_rvv_grid
from .simulation import run_simulation
from .snapshots import init_granting, init_grid, pickle_mass_import, pickle_mass_zip

# Colonnes exportees dans le snapshot performing de decembre (cf. notebook cellule finale).
PERF_DEC_COLUMNS = [
    "Contract_ID", "Age", "Nombre_enfant", "Situation_matrimoniale", "situation_habitat",
    "CSP", "anc_emp_continuous", "plafond_credit", "ResidualMaturity", "Date", "Retard",
    "Retard_cumule", "Unemp", "dec_flag", "dispo_ratio", "draw_senior", "banking_seniority",
    "Closed", "Disposable", "montant_credit", "NbFac", "performing_claim", "Outstanding",
    "Principal", "Charges_ajuste", "Revenu_ajuste", "ScorePD", "ScoreRVV", "Default12m",
]


@dataclass
class RunResult:
    """Resume d'un run : chemins des principaux exports et quelques compteurs."""

    version: str
    output_dir: str
    granting_csv: str
    score_grid_pd_csv: str
    score_grid_rvv_csv: str
    complete_csv: str
    full_csv: str
    perf_dec_csv: str
    archive_zip: str
    n_contracts: int
    n_rows_full: int
    timings: dict          # duree (s) par grande etape
    total_seconds: float   # somme des durees chronometrees


def _set_seeds(seed: int) -> None:
    """Fixe les graines aleatoires globales pour rendre le run reproductible."""
    np.random.seed(seed)
    random.seed(seed)


def run(cfg: SimulationConfig | None = None) -> RunResult:
    """Execute la simulation complete et retourne un resume des exports produits."""
    cfg = cfg or SimulationConfig()
    cfg.ensure_dirs()

    # Affichage complet des DataFrames + suppression des avertissements (comme le notebook).
    pd.options.display.max_columns = None
    warnings.filterwarnings("ignore")

    _set_seeds(cfg.seed)

    with Reporter(cfg.log_path, cfg.controls_dir) as reporter:
        reporter.section(f"Simulation RDS PD - version {cfg.version}")
        reporter.log(f"Contrats : {cfg.n_contracts} | rho : {cfg.rho} | graine : {cfg.seed}")
        reporter.log(f"Entrees : {cfg.inputs_dir}")
        reporter.log(f"Sorties : {cfg.version_dir}")

        # 1) Octroi des contrats -> 00_Granting.csv
        with reporter.timed("Octroi des contrats"):
            run_granting(cfg, reporter)

        # 2) Grilles de score (exportees en CSV, puis relues pour le round-trip de types)
        with reporter.timed("Grilles de score"):
            reporter.section("Construction des grilles de score")
            build_rvv_grid(cfg)
            build_pd_grid(cfg, reporter)

        # 3) Initialisation (relit granting + grilles depuis les CSV exportes)
        with reporter.timed("Initialisation"):
            data = init_granting(cfg)
            score_grid_pd, score_grid_rvv = init_grid(cfg)
            macro = load_macro(cfg)

        # 4) Boucle de simulation mensuelle (ecrit les pickles)
        with reporter.timed("Simulation mensuelle"):
            run_simulation(cfg, data, score_grid_pd, score_grid_rvv, macro, reporter)

        # 5) Consolidation : import des pickles + flag de defaut a 12 mois
        with reporter.timed("Consolidation"):
            reporter.section("Consolidation et exports")
            data = pickle_mass_import(cfg.snapshots_dir)
            data = data.reset_index().sort_values(by=["Contract_ID", "Facility_ID", "Date"])

            defaulted = data.query("Master & (Defaut==1)")[["Contract_ID", "Date"]]
            defaulted.columns = ["Contract_ID", "Default_dt"]
            data = data.merge(defaulted, on="Contract_ID", how="left")
            data["Default12m"] = np.where(
                (data.Date < data.Default_dt) & (data.Date + pd.DateOffset(months=12) >= data.Default_dt), 1, 0
            )
            complete_csv = cfg.output_path("RDSPD_CompleteMaster&Fac.csv")
            data.to_csv(complete_csv, encoding="utf-8")
            reporter.show_df(data.head(5), "complete_master_fac_head", title="Apercu master & facilites :")

        # 6) Controles post-simulation (grille relue depuis le CSV)
        with reporter.timed("Controles de validation"):
            grid = read_csv(cfg.output_path("ScoreGrid_PD.csv"))
            controls.control_score_mapping(data, grid, reporter)
            control_frame = controls.prepare_control_frame(data)
            controls.control_p0p1p2_pdt(control_frame, grid, reporter)
            controls.control_pd12(control_frame, grid, reporter)

        # 7) Exports finaux
        with reporter.timed("Exports finaux"):
            full_csv = cfg.output_path(f"RDSPD_Full_{cfg.version}.csv")
            data.to_csv(full_csv, index=True, encoding="utf-8")
            reporter.log(f"Obs Full RDS -  {len(data)}")
            reporter.log(f"    > Unique Contract_ID;  {data.Contract_ID.nunique()}")

            data_filtered = data.query("Master & Defaut==0 & Month==12")[PERF_DEC_COLUMNS]
            reporter.log(f"Obs Filtered -  {len(data_filtered)}")
            reporter.log(f"    > Unique Contract_ID;  {data_filtered.Contract_ID.nunique()}")
            perf_dec_csv = cfg.output_path(f"RDSPD_Perf_Dec_{cfg.version}.csv")
            data_filtered.to_csv(perf_dec_csv, index=False, encoding="utf-8")

        # 8) Archivage des pickles
        with reporter.timed("Archivage des pickles"):
            pickle_mass_zip(cfg.snapshots_dir, cfg.archive_path)
            reporter.log(f"Archive des snapshots : {cfg.archive_path}")

        # Recapitulatif des temps d'execution
        reporter.timing_summary()

        reporter.section("Termine")
        reporter.log(f"Tous les exports sont dans : {cfg.version_dir}")

        return RunResult(
            version=cfg.version,
            output_dir=str(cfg.version_dir),
            granting_csv=str(cfg.output_path("00_Granting.csv")),
            score_grid_pd_csv=str(cfg.output_path("ScoreGrid_PD.csv")),
            score_grid_rvv_csv=str(cfg.output_path("ScoreGrid_RVV.csv")),
            complete_csv=str(complete_csv),
            full_csv=str(full_csv),
            perf_dec_csv=str(perf_dec_csv),
            archive_zip=str(cfg.archive_path),
            n_contracts=cfg.n_contracts,
            n_rows_full=len(data),
            timings=reporter.timings,
            total_seconds=sum(reporter.timings.values()),
        )
