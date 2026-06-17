"""Utilitaires d'entrees/sorties.

Le point essentiel ici est la lecture robuste des CSV : plusieurs fichiers d'entree
(``RDSPD_ProbaCSP_Conso.csv``, ``RDSPD_IncomeIndex.csv``, ``RDSPD_ProbaHousing_Conso.csv``)
sont encodes en **cp1252** et non en UTF-8. Or le scorecard PD compare des libelles CSP
accentues en litteral (``'Independants'``, ``'Employes'``, ``'Professions intermediaires'``...).
Une lecture en UTF-8 corromprait silencieusement ces chaines et ferait echouer le scoring
(toutes les modalites tomberaient en ``'Error'``). On lit donc en essayant successivement
UTF-8 puis cp1252/latin-1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Ordre d'essai des encodages. UTF-8 d'abord (avec BOM eventuel), puis cp1252/latin-1
# qui couvrent les accents francais des fichiers exportes depuis un tableur Windows.
_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Lit un CSV en gerant automatiquement l'encodage (UTF-8 puis cp1252/latin-1).

    Parameters
    ----------
    path:
        Chemin du fichier CSV.
    **kwargs:
        Arguments transmis tels quels a :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        Le contenu du fichier, avec les accents correctement decodes.
    """
    path = Path(path)
    last_error: Exception | None = None
    for encoding in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:  # pragma: no cover - depend des fichiers
            last_error = exc
            continue
    # latin-1 ne devrait jamais echouer ; ce point n'est atteint qu'en cas de probleme reel.
    raise UnicodeDecodeError(  # pragma: no cover
        "rdspd_core",
        b"",
        0,
        1,
        f"Impossible de decoder {path} avec {_ENCODINGS}: {last_error}",
    )


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    """Ecrit un DataFrame en CSV (UTF-8) en creant le repertoire parent si besoin."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8", **kwargs)
    return path
