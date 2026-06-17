"""Generateur de donnees non normales (systeme de Pearson).

Package ``RandNonNormal`` vendorise depuis le projet d'origine (auteur : nadahannad,
2023). Il fournit la fonction :func:`IG` (Independent Generator) utilisee pour generer
des donnees multivariees respectant une matrice de covariance, une asymetrie (skewness)
et un aplatissement (kurtosis) cibles. Le comportement numerique est strictement
identique a la version d'origine ; seuls les imports ont ete remontes en tete de module.
"""

from __future__ import annotations

from .generator import IG

__all__ = ["IG"]
