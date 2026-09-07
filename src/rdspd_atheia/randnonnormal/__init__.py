"""Non-normal random data generation through the Pearson distribution system.

Exposes :func:`IG` (Independent Generator), which draws multivariate data matching a
target covariance matrix together with per-variable skewness and excess kurtosis.

The implementation follows published mathematics (Pearson system; Heinrich 2004 for
the type IV sampler; Vale & Maurelli 1983 for the independent-generator construction).
See the NOTICE file at the repository root for the full references.
"""

from __future__ import annotations

from .generator import IG, fit_pearson_from_moments

__all__ = ["IG", "fit_pearson_from_moments"]
