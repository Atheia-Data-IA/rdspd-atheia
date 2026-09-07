"""Fixtures partagees pour les tests."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = PROJECT_ROOT / "inputs" / "parameters"
REFERENCE_DIR = PROJECT_ROOT / "inputs" / "reference"


@pytest.fixture(autouse=True)
def _require_inputs() -> None:
    """Saute les tests si les donnees d'entree locales sont absentes (clone code seul)."""
    if not INPUTS_DIR.exists():
        pytest.skip("Donnees d'entree absentes (dossier inputs/ non present)")


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def reference_dir() -> Path:
    if not REFERENCE_DIR.exists():
        pytest.skip("Repertoire de reference absent")
    return REFERENCE_DIR
