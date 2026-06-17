"""Affichage et archivage des sorties de controle.

Le notebook melait ``print(...)`` (affichage texte) et ``display(...)`` (rendu Jupyter
des DataFrames). En script, on remplace tout par un :class:`Reporter` unique qui, pour
chaque sortie :

- l'**affiche** sur la console (exigence « les sorties doivent etre printees a l'execution ») ;
- l'**ecrit** dans ``outputs/<version>/`` (log de run + un fichier par controle) ;

ce qui satisfait l'exigence « ... mais figurer egalement dans le dossier outputs ».
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class Reporter:
    """Diffuse les messages et tableaux de controle vers la console et le disque."""

    def __init__(self, log_path: str | Path, controls_dir: str | Path) -> None:
        self.log_path = Path(log_path)
        self.controls_dir = Path(controls_dir)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.controls_dir.mkdir(parents=True, exist_ok=True)
        # Reinitialise le log a chaque run.
        self._log_handle = self.log_path.open("w", encoding="utf-8")

    # -- gestion du cycle de vie ------------------------------------------------
    def close(self) -> None:
        if not self._log_handle.closed:
            self._log_handle.close()

    def __enter__(self) -> Reporter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- messages texte ---------------------------------------------------------
    def log(self, message: str = "") -> None:
        """Affiche un message et l'ajoute au log de run."""
        print(message)
        self._log_handle.write(str(message) + "\n")
        self._log_handle.flush()

    def section(self, title: str) -> None:
        """Affiche un titre de section encadre."""
        bar = "=" * 70
        self.log("")
        self.log(bar)
        self.log(f"  {title}")
        self.log(bar)

    # -- tableaux (DataFrame / Series) -----------------------------------------
    def show_df(
        self,
        df: pd.DataFrame | pd.Series,
        name: str,
        title: str | None = None,
        save: bool = True,
    ) -> None:
        """Affiche un tableau et, optionnellement, l'enregistre en CSV dans ``controls/``.

        Parameters
        ----------
        df:
            DataFrame ou Series a afficher.
        name:
            Nom de base du fichier CSV (sans extension) ecrit dans ``controls/``.
        title:
            Titre facultatif affiche avant le tableau.
        save:
            Si vrai, ecrit le tableau en CSV.
        """
        if title:
            self.log(title)
        self.log(df.to_string())
        if save:
            out = self.controls_dir / f"{name}.csv"
            frame = df.to_frame() if isinstance(df, pd.Series) else df
            frame.to_csv(out, encoding="utf-8")
