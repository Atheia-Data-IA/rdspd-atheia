"""Programme principal : boucle de simulation mensuelle des snapshots.

Reproduit la cellule « Main program » du notebook. Pour chaque fin de mois, on construit
ou met a jour le snapshot des contrats et des facilites, on applique les scores et on
serialise le resultat dans un pickle.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from .config import SimulationConfig
from .macro import MacroData
from .reporting import Reporter
from .snapshots import (
    append_snapshot_ctr,
    append_snapshot_fac,
    increment_snapshot_ctr,
    init_snapshot_ctr,
    init_snapshot_fac,
    origin_fac,
    snap_pickle_export,
    update_default_ctr,
    update_financials_all,
    update_snapshot_ctr,
    update_snapshot_fac,
)


@dataclass
class SimContext:
    """Etat partage par la boucle de simulation (remplace les variables globales du notebook)."""

    cfg: SimulationConfig
    data: pd.DataFrame
    score_grid_pd: pd.DataFrame
    score_grid_rvv: pd.DataFrame
    macro: MacroData
    start: pd.Timestamp


def build_date_list(cfg: SimulationConfig) -> pd.DatetimeIndex:
    """Plage de dates (fins de mois) a traiter."""
    return pd.date_range(start=cfg.start, end=cfg.end, freq=cfg.freq)


def loop_snapshot(dt: pd.Timestamp, ctx: SimContext) -> None:
    """Traite un snapshot mensuel et le serialise dans un pickle."""
    cfg = ctx.cfg

    # Contrats - initialisation ou increment
    if dt == ctx.start:
        snpctr = init_snapshot_ctr(ctx.data, dt)  # nouveaux contrats octroyes
        ptffac = pd.DataFrame([])
    else:
        ptfctr, newctr, ptffac = increment_snapshot_ctr(ctx.data, dt, cfg.snapshots_dir)
        ptfctr = update_snapshot_ctr(ptfctr, dt, ctx.macro, ctx.score_grid_rvv)
        snpctr = append_snapshot_ctr(ptfctr, newctr)

    # Facilites - increment
    newfac = origin_fac(snpctr, ptffac)
    newfac = init_snapshot_fac(newfac, dt, ctx.macro.interest_rates)
    snpfac = append_snapshot_fac(ptffac, newfac)
    snpfac = update_snapshot_fac(snpfac, dt)

    # Contrats - agregats et score PD
    snpctr, snpfac = update_default_ctr(snpctr, snpfac, ctx.score_grid_pd, cfg.rho, ctx.macro.aggregate)

    # Etats financiers et statut de cloture
    snpctr, snpfac = update_financials_all(snpctr, snpfac, cfg.fees)

    # Serialisation
    snap_pickle_export(dt, snpctr, snpfac, cfg.snapshots_dir)


def run_simulation(
    cfg: SimulationConfig,
    data: pd.DataFrame,
    score_grid_pd: pd.DataFrame,
    score_grid_rvv: pd.DataFrame,
    macro: MacroData,
    reporter: Reporter,
) -> SimContext:
    """Execute la boucle de simulation sur toute la plage de dates."""
    reporter.section("Simulation des snapshots mensuels")

    date_list = build_date_list(cfg)
    start = date_list[0]
    ctx = SimContext(
        cfg=cfg,
        data=data,
        score_grid_pd=score_grid_pd,
        score_grid_rvv=score_grid_rvv,
        macro=macro,
        start=start,
    )

    reporter.log(f"Plage : {date_list[0].date()} -> {date_list[-1].date()}  ({len(date_list)} snapshots)")
    for dt in tqdm(date_list, desc="Processing IDs"):
        loop_snapshot(dt, ctx)

    return ctx
