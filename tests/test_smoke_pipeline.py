"""Smoke test : run reduit de bout-en-bout et verification des exports.

Le run complet (50 000 contrats x 216 mois) etant tres long, on valide ici le pipeline
sur un petit portefeuille et une plage de dates courte.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rdspd_atheia.config import SimulationConfig
from rdspd_atheia.io_utils import read_csv
from rdspd_atheia.pipeline import run

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
        "run.log",
    ]
    for name in expected:
        assert (out / name).exists(), f"export manquant : {name}"

    # Fichiers redondants supprimes : le doublon CompleteMaster&Fac et l'archive de pickles.
    assert not (out / "RDSPD_CompleteMaster&Fac.csv").exists()
    assert not (out / "archive_pickle.zip").exists()
    # Pickles mensuels supprimes apres consolidation.
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


def _small_cfg(project_root, outputs_dir, version, **kw):
    return SimulationConfig(
        project_root=project_root,
        outputs_dir=outputs_dir,
        version=version,
        n_contracts=200,
        start="2006-01-31",
        end="2006-03-31",
        seed=42,
        **kw,
    )


def test_clean_outputs_removes_stale_files(project_root, tmp_path):
    """Par defaut, outputs/<version>/ est purge : les artefacts d'un run precedent disparaissent."""
    cfg = _small_cfg(project_root, tmp_path, "CLEAN")
    cfg.version_dir.mkdir(parents=True, exist_ok=True)
    stale = cfg.version_dir / "RDSPD_CompleteMaster&Fac.csv"  # fichier d'un ancien format
    stale.write_text("obsolete", encoding="utf-8")

    result = run(cfg)

    assert not stale.exists(), "le fichier perime aurait du etre supprime"
    assert (Path(result.output_dir) / "00_Granting.csv").exists()


def test_no_clean_preserves_existing_files(project_root, tmp_path):
    """Avec clean_outputs=False (--no-clean), l'existant est conserve."""
    cfg = _small_cfg(project_root, tmp_path, "KEEP", clean_outputs=False)
    cfg.version_dir.mkdir(parents=True, exist_ok=True)
    stale = cfg.version_dir / "vieux_fichier.csv"
    stale.write_text("conserve", encoding="utf-8")

    run(cfg)

    assert stale.exists(), "le fichier existant aurait du etre conserve"
    assert stale.read_text(encoding="utf-8") == "conserve"


def test_clean_version_dir_refuse_hors_outputs(project_root, tmp_path):
    """Garde-fou : refuse de purger si version_dir n'est pas un sous-repertoire de outputs_dir."""
    cfg = SimulationConfig(project_root=project_root, outputs_dir=tmp_path, version=".")
    with pytest.raises(ValueError, match="Refus de purger"):
        cfg.clean_version_dir()


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
