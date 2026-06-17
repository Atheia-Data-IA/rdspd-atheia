"""Configuration de la simulation RDS PD.

Toute la parametrisation du notebook d'origine (version, taille du portefeuille,
plage de dates, parametre d'autocorrelation des actifs, graines aleatoires, chemins
d'entree/sortie, scorecards utilisees) est centralisee dans :class:`SimulationConfig`.

Aucun chemin n'est code en dur dans la logique metier : tout passe par cet objet,
ce qui permettra a une future API de construire une configuration et de lancer un run
sans modifier le code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# Sous-dossier de ``inputs/`` reproduisant l'arborescence du Google Drive d'origine.
# Le notebook lisait ses entrees dans ``.../My Drive/01_Sujets_Internes/00_RDS_PD/Google_Sheets/``.
DEFAULT_INPUT_SUBDIR = Path("01_Sujets_Internes") / "00_RDS_PD" / "Google_Sheets"


@dataclass
class SimulationConfig:
    """Parametres d'un run de simulation.

    Les valeurs par defaut reproduisent exactement celles du notebook
    ``SimulationRDSPD_V1.3`` (version ``PD02R50k0623TTC``, 50 000 contrats,
    snapshots mensuels de janvier 2006 a decembre 2023).
    """

    # --- Identifiant du run ----------------------------------------------------
    version: str = "PD02R50k0623TTC"

    # --- Generation des contrats ----------------------------------------------
    n_contracts: int = 50_000          # Taille de la matrice generee (N du notebook)
    reps: int = 1                      # Nombre de simulations (reps du notebook)
    type_a: str = "symm"              # Type de la matrice de covariance ('symm' ou 'triang')
    reference_year: int = 2021         # Annee de reference pour skewness / kurtosis

    # --- Plage de snapshots ----------------------------------------------------
    start: str = "2006-01-31"
    end: str = "2023-12-31"
    freq: str = "ME"                  # Fin de mois (Month-End)

    # --- Parametres de risque --------------------------------------------------
    rho: float = 0.05                  # Autocorrelation des actifs (impact sur P0 conditionnel)
    fees: int = 200                    # Frais forfaitaires sur impayes (en centimes)

    # --- Graines aleatoires ----------------------------------------------------
    # NOUVEAU : graine globale pour rendre les runs reproductibles entre eux.
    # (Le notebook n'en fixait pas, ses sorties stochastiques variaient a chaque execution.)
    seed: int = 42
    # Graine interne preservee a l'identique du notebook (fonction destandarization).
    destandarization_seed: int = 42

    # --- Scorecards (fichiers RDSPD_PartialScore_<id>.csv / RDSPD_ScoreParams_<id>.csv) ---
    scorecard_pd: str = "PD02R3120"
    scorecard_rvv: str = "RVV00R0030"

    # --- Chemins ---------------------------------------------------------------
    project_root: Path = field(default_factory=Path.cwd)
    inputs_dir: Path | None = None     # defaut : <project_root>/inputs/<DEFAULT_INPUT_SUBDIR>
    outputs_dir: Path | None = None    # defaut : <project_root>/outputs

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        if self.inputs_dir is None:
            self.inputs_dir = self.project_root / "inputs" / DEFAULT_INPUT_SUBDIR
        else:
            self.inputs_dir = Path(self.inputs_dir)
        if self.outputs_dir is None:
            self.outputs_dir = self.project_root / "outputs"
        else:
            self.outputs_dir = Path(self.outputs_dir)

    # --- Repertoires de sortie -------------------------------------------------
    @property
    def version_dir(self) -> Path:
        """Repertoire de sortie du run : ``outputs/<version>/``."""
        return self.outputs_dir / self.version

    @property
    def snapshots_dir(self) -> Path:
        """Repertoire des pickles de snapshots mensuels."""
        return self.version_dir / "snapshots"

    @property
    def controls_dir(self) -> Path:
        """Repertoire des rapports de controle de validation."""
        return self.version_dir / "controls"

    @property
    def log_path(self) -> Path:
        """Fichier de log du run."""
        return self.version_dir / "run.log"

    @property
    def archive_path(self) -> Path:
        """Archive zip des pickles de snapshots."""
        return self.version_dir / "archive_pickle.zip"

    # --- Helpers de chemins ----------------------------------------------------
    def input_path(self, filename: str) -> Path:
        """Chemin complet d'un fichier d'entree."""
        return self.inputs_dir / filename

    def output_path(self, filename: str) -> Path:
        """Chemin complet d'un fichier de sortie (dans ``outputs/<version>/``)."""
        return self.version_dir / filename

    def ensure_dirs(self) -> None:
        """Cree les repertoires de sortie necessaires au run."""
        for d in (self.version_dir, self.snapshots_dir, self.controls_dir):
            d.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **kwargs) -> SimulationConfig:
        """Retourne une copie de la configuration avec les champs donnes remplaces."""
        return replace(self, **kwargs)


# Surcharges pour un run rapide de bout-en-bout (smoke test).
SMOKE_OVERRIDES: dict = {
    "version": "SMOKE_PD02R50k0623TTC",
    "n_contracts": 2_000,
    "start": "2006-01-31",
    "end": "2007-12-31",
}


def smoke_config(**kwargs) -> SimulationConfig:
    """Configuration reduite pour un run rapide (petit N, plage de dates courte)."""
    params: dict = dict(SMOKE_OVERRIDES)
    params.update(kwargs)
    return SimulationConfig(**params)
