"""Interface en ligne de commande de la simulation RDS PD.

Exemples
--------
    rdspd-simulate                 # run complet (50 000 contrats, 2006-2023)
    rdspd-simulate --smoke         # run rapide reduit (smoke test)
    rdspd-simulate --n-contracts 5000 --end 2010-12-31 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import SimulationConfig, smoke_config
from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments de la CLI."""
    p = argparse.ArgumentParser(
        prog="rdspd-simulate",
        description="Simulateur de portefeuille de credit synthetique RDS PD.",
    )
    p.add_argument("--smoke", action="store_true", help="Run rapide reduit (petit N, plage courte) pour validation.")
    p.add_argument("--version", dest="version", help="Identifiant de version / run.")
    p.add_argument("--n-contracts", type=int, help="Nombre de contrats a generer.")
    p.add_argument("--start", help="Date de debut (YYYY-MM-DD).")
    p.add_argument("--end", help="Date de fin (YYYY-MM-DD).")
    p.add_argument("--rho", type=float, help="Autocorrelation des actifs (impact sur P0).")
    p.add_argument("--seed", type=int, help="Graine aleatoire globale (reproductibilite).")
    p.add_argument("--inputs-dir", type=Path, help="Repertoire des fichiers d'entree.")
    p.add_argument("--outputs-dir", type=Path, help="Repertoire des sorties.")
    p.add_argument("--scorecard-pd", help="Identifiant du scorecard PD (ex. PD02R3120).")
    p.add_argument("--scorecard-rvv", help="Identifiant du scorecard revolving (ex. RVV00R0030).")
    return p


def _config_from_args(args: argparse.Namespace) -> SimulationConfig:
    """Construit la configuration a partir des arguments (le preset smoke sert de base)."""
    overrides = {
        "version": args.version,
        "n_contracts": args.n_contracts,
        "start": args.start,
        "end": args.end,
        "rho": args.rho,
        "seed": args.seed,
        "inputs_dir": args.inputs_dir,
        "outputs_dir": args.outputs_dir,
        "scorecard_pd": args.scorecard_pd,
        "scorecard_rvv": args.scorecard_rvv,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if args.smoke:
        return smoke_config(**overrides)
    return SimulationConfig(**overrides)


def main(argv: list[str] | None = None) -> int:
    """Point d'entree console."""
    args = build_parser().parse_args(argv)
    cfg = _config_from_args(args)
    result = run(cfg)
    print("")
    print(f"Run termine : {result.version}")
    print(f"  Exports    : {result.output_dir}")
    print(f"  Lignes RDS Full : {result.n_rows_full}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
