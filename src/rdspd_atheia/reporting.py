"""Affichage et archivage des sorties de controle.

Le notebook melait ``print(...)`` (affichage texte) et ``display(...)`` (rendu Jupyter
des DataFrames). En script, on remplace tout par un :class:`Reporter` unique qui, pour
chaque sortie :

- l'**affiche** sur la console (exigence « les sorties doivent etre printees a l'execution ») ;
- l'**ecrit** dans ``outputs/<version>/`` (log de run + un fichier par controle) ;

ce qui satisfait l'exigence « ... mais figurer egalement dans le dossier outputs ».
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


def format_duration(seconds: float) -> str:
    """Formate une duree en texte lisible (ASCII uniquement pour la console Windows)."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        minutes, sec = divmod(seconds, 60)
        return f"{int(minutes)} min {sec:04.1f} s"
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{int(hours)} h {int(minutes):02d} min {int(sec):02d} s"


class Reporter:
    """Diffuse les messages et tableaux de controle vers la console et le disque."""

    def __init__(self, log_path: str | Path, controls_dir: str | Path) -> None:
        self.log_path = Path(log_path)
        self.controls_dir = Path(controls_dir)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.controls_dir.mkdir(parents=True, exist_ok=True)
        # Reinitialise le log a chaque run.
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        # Chronometrage des grandes etapes : liste de (label, duree_en_secondes).
        self._timings: list[tuple[str, float]] = []

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

    # -- chronometrage des etapes ----------------------------------------------
    @contextmanager
    def timed(self, label: str):
        """Chronometre une grande etape et affiche sa duree a la fin.

        Utilisation : ``with reporter.timed("Octroi"): ...``. La duree est affichee
        dans le terminal (et le log), puis conservee pour le recapitulatif final.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._timings.append((label, dt))
            self.log(f"  [temps] {label} : {format_duration(dt)}")

    @property
    def timings(self) -> dict[str, float]:
        """Durees mesurees par etape (en secondes)."""
        return dict(self._timings)

    def timing_summary(self) -> None:
        """Affiche et enregistre le recapitulatif des temps d'execution par etape."""
        if not self._timings:
            return
        total = sum(d for _, d in self._timings)
        self.section("Temps d'execution par etape")
        width = max(len(label) for label, _ in self._timings)
        for label, dt in self._timings:
            pct = 100 * dt / total if total else 0.0
            self.log(f"  {label:<{width}}  {format_duration(dt):>14}  ({pct:5.1f} %)")
        self.log(f"  {'TOTAL':<{width}}  {format_duration(total):>14}  (100.0 %)")
        # Enregistrement durable dans controls/timings.csv
        pd.DataFrame(self._timings, columns=["etape", "duree_s"]).to_csv(
            self.controls_dir / "timings.csv", index=False, encoding="utf-8"
        )

    # -- tableaux (DataFrame / Series) -----------------------------------------
    def show_df(
        self,
        df: pd.DataFrame | pd.Series,
        name: str,
        title: str | None = None,
        save: bool = True,
        preview_cols: Sequence[str] | None = None,
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
        preview_cols:
            Sous-ensemble de colonnes a afficher sur la console : un tableau de plusieurs
            dizaines de colonnes est illisible dans un terminal (lignes repliees). N'affecte
            que l'affichage -- le CSV ecrit dans ``controls/`` conserve toutes les colonnes.
        """
        if title:
            self.log(title)

        shown = df
        n_total = df.shape[1] if isinstance(df, pd.DataFrame) else 1
        if preview_cols is not None and isinstance(df, pd.DataFrame):
            kept = [c for c in preview_cols if c in df.columns]
            if kept and len(kept) < n_total:
                shown = df[kept]

        self.log(shown.to_string())
        if shown is not df:
            detail = f" - tableau complet : controls/{name}.csv" if save else ""
            self.log(f"  (apercu : {shown.shape[1]} des {n_total} colonnes{detail})")

        if save:
            out = self.controls_dir / f"{name}.csv"
            frame = df.to_frame() if isinstance(df, pd.Series) else df
            frame.to_csv(out, encoding="utf-8")
