"""Verifie que les grilles de score generees correspondent aux references.

Les grilles RVV et PD sont entierement deterministes (produit cartesien + optimisation
deterministe). Elles constituent l'ancre de validation du portage : elles doivent
reproduire les fichiers de reference produits par le notebook.
"""

from __future__ import annotations

import numpy as np

from rdspd_core.config import SimulationConfig
from rdspd_core.io_utils import read_csv
from rdspd_core.scores import build_pd_grid, build_rvv_grid

RVV_KEYS = ["dec_flag", "dispo_ratio_bin", "draw_senior_bin"]
PD_KEYS = ["rd_Age_AncEmp", "rd_DebtRatioW", "situation_habitat", "rd_UsingRate", "rd_CSP", "rd_BankingSen"]


def _make_cfg(project_root, tmp_path) -> SimulationConfig:
    cfg = SimulationConfig(project_root=project_root, outputs_dir=tmp_path)
    cfg.ensure_dirs()
    return cfg


def test_rvv_grid_matches_reference(project_root, reference_dir, tmp_path):
    cfg = _make_cfg(project_root, tmp_path)
    build_rvv_grid(cfg)

    got = read_csv(cfg.output_path("ScoreGrid_RVV.csv")).sort_values(RVV_KEYS).reset_index(drop=True)
    ref = read_csv(reference_dir / "ScoreGrid_RVV.csv").sort_values(RVV_KEYS).reset_index(drop=True)

    assert got.shape == ref.shape == (32, 11)
    for k in RVV_KEYS:
        assert list(got[k].astype(str)) == list(ref[k].astype(str))
    # Toutes les colonnes numeriques sont des formules pures -> egalite quasi exacte.
    for col in ["ScoreRVV", "Odds", "RVV12", "RVVt"]:
        np.testing.assert_allclose(got[col].to_numpy(), ref[col].to_numpy(), atol=1e-9, rtol=1e-9)


def test_pd_grid_matches_reference(project_root, reference_dir, tmp_path):
    cfg = _make_cfg(project_root, tmp_path)
    build_pd_grid(cfg)

    got = read_csv(cfg.output_path("ScoreGrid_PD.csv")).sort_values(PD_KEYS).reset_index(drop=True)
    ref = read_csv(reference_dir / "ScoreGrid_PD.csv").sort_values(PD_KEYS).reset_index(drop=True)

    assert got.shape == ref.shape == (3750, 20)
    for k in PD_KEYS:
        assert list(got[k].astype(str)) == list(ref[k].astype(str))

    # Colonnes deterministes (formules pures) : egalite quasi exacte.
    for col in ["ScorePD", "Odds", "PD12", "PDt"]:
        np.testing.assert_allclose(got[col].to_numpy(), ref[col].to_numpy(), atol=1e-9, rtol=1e-9)

    # Decomposition optimisee (P0, P1, P2) : reproduit la reference.
    for col in ["P0", "P1", "P2"]:
        np.testing.assert_allclose(got[col].to_numpy(), ref[col].to_numpy(), atol=1e-6)

    # Contrainte du modele : P0 * P1 * P2 == PDt.
    prod = (got.P0 * got.P1 * got.P2).to_numpy()
    np.testing.assert_allclose(prod, got.PDt.to_numpy(), atol=1e-5)
