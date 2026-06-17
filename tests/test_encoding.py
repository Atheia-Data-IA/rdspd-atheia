"""Verifie que la lecture CSV decode correctement les fichiers cp1252.

Les libelles CSP / habitat accentues sont compares en litteral par le scorecard PD :
un mauvais decodage casserait silencieusement le scoring.
"""

from __future__ import annotations

from rdspd_core.io_utils import read_csv


def test_csp_labels_decoded(project_root):
    base = project_root / "inputs" / "01_Sujets_Internes" / "00_RDS_PD" / "Google_Sheets"
    df = read_csv(base / "RDSPD_ProbaCSP_Conso.csv")
    cols = set(df.columns)
    assert "Indépendants" in cols
    assert "Employés" in cols
    assert "Professions intermédiaires" in cols


def test_housing_labels_decoded(project_root):
    base = project_root / "inputs" / "01_Sujets_Internes" / "00_RDS_PD" / "Google_Sheets"
    df = read_csv(base / "RDSPD_ProbaHousing_Conso.csv")
    cols = set(df.columns)
    assert "Propriétaire" in cols
    assert "Non Propriétaire" in cols


def test_income_index_csp_decoded(project_root):
    base = project_root / "inputs" / "01_Sujets_Internes" / "00_RDS_PD" / "Google_Sheets"
    df = read_csv(base / "RDSPD_IncomeIndex.csv")
    cols = set(df.columns)
    assert "Employés" in cols
    assert "Indépendants" in cols
