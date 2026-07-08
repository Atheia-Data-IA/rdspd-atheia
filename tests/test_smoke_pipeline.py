"""Smoke test : run reduit de bout-en-bout et verification des exports.

Le run complet (50 000 contrats x 216 mois) etant tres long, on valide ici le pipeline
sur un petit portefeuille et une plage de dates courte.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rdspd_core.config import SimulationConfig
from rdspd_core.io_utils import read_csv
from rdspd_core.pipeline import run

PERF_DEC_NCOLS = 29
GRANTING_NCOLS = 20


def test_smoke_run_parquet_default(project_root, tmp_path):
    """Run par defaut : exports finaux en Parquet, plus de CompleteMaster&Fac."""
    cfg = SimulationConfig(
        project_root=project_root,
        outputs_dir=tmp_path,
        version="SMOKE_TEST",
        n_contracts=250,
        start="2006-01-31",
        end="2006-12-31",
        seed=42,
    )
    result = run(cfg)
    assert result.export_format == "parquet"

    out = Path(result.output_dir)
    expected = [
        "00_Granting.csv",  # intermediaire relu -> reste CSV
        "ScoreGrid_RVV.csv",
        "ScoreGrid_PD.csv",
        f"RDSPD_Full_{cfg.version}.parquet",
        f"RDSPD_Perf_Dec_{cfg.version}.parquet",
        "archive_pickle.zip",
        "run.log",
    ]
    for name in expected:
        assert (out / name).exists(), f"export manquant : {name}"

    # Fichier redondant supprime.
    assert not (out / "RDSPD_CompleteMaster&Fac.csv").exists()
    # Pickles archives puis supprimes.
    assert not list((out / "snapshots").glob("*.pkl"))
    # Controles + recap des temps ecrits.
    assert (out / "controls" / "timings.csv").exists()

    # Schemas attendus (Perf_Dec relu en Parquet).
    assert len(read_csv(out / "00_Granting.csv").columns) == GRANTING_NCOLS
    assert len(pd.read_parquet(out / f"RDSPD_Perf_Dec_{cfg.version}.parquet").columns) == PERF_DEC_NCOLS
    assert result.n_rows_full > 0


def test_smoke_run_csv_format(project_root, tmp_path):
    """--exportformat csv : exports finaux en CSV avec le meme schema."""
    cfg = SimulationConfig(
        project_root=project_root,
        outputs_dir=tmp_path,
        version="SMOKE_CSV",
        n_contracts=250,
        start="2006-01-31",
        end="2006-12-31",
        seed=42,
        export_format="csv",
    )
    result = run(cfg)
    assert result.export_format == "csv"

    out = Path(result.output_dir)
    assert (out / f"RDSPD_Full_{cfg.version}.csv").exists()
    assert (out / f"RDSPD_Perf_Dec_{cfg.version}.csv").exists()
    assert not (out / f"RDSPD_Full_{cfg.version}.parquet").exists()
    assert len(read_csv(out / f"RDSPD_Perf_Dec_{cfg.version}.csv").columns) == PERF_DEC_NCOLS


def test_smoke_run_is_reproducible(project_root, tmp_path):
    """Deux runs avec la meme graine produisent un octroi identique."""
    common = dict(
        project_root=project_root,
        n_contracts=200,
        start="2006-01-31",
        end="2006-03-31",
        seed=123,
    )
    cfg1 = SimulationConfig(outputs_dir=tmp_path / "a", version="REPRO_A", **common)
    cfg2 = SimulationConfig(outputs_dir=tmp_path / "b", version="REPRO_B", **common)
    run(cfg1)
    run(cfg2)

    g1 = read_csv(cfg1.output_path("00_Granting.csv"))
    g2 = read_csv(cfg2.output_path("00_Granting.csv"))
    # Colonnes numeriques de l'octroi : strictement identiques a graine egale.
    for col in ["Age", "Revenu", "Charges", "plafond_credit", "Maturite", "banking_seniority"]:
        assert list(g1[col]) == list(g2[col]), f"colonne non reproductible : {col}"
