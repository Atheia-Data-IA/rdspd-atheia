"""Sampling from the Pearson distribution system, and the Independent Generator.

The public entry point is :func:`IG`, which draws multivariate data reproducing a
target covariance matrix together with a per-variable skewness and excess kurtosis.

Method and sources
------------------
The mathematics implemented here are public:

* the Pearson family and its classification from the first four moments
  (K. Pearson, 1895; Johnson, Kotz & Balakrishnan, *Continuous Univariate
  Distributions*, vol. 1, ch. 12). The type is selected from the discriminant
  ``kappa = b1 (b2 + 3)^2 / (4 (4 b2 - 3 b1)(2 b2 - 3 b1 - 6))``;
* the rejection sampler for the Pearson type IV density, and its normalising
  constant expressed through ``|Gamma(m + i nu/2) / Gamma(m)|^2``, follow
  J. Heinrich, *A Guide to the Pearson Type IV Distribution* (2004);
* the "independent generator" construction -- solving for univariate third and
  fourth moments such that a fixed linear combination reproduces the target
  multivariate moments -- follows the Vale & Maurelli (1983) family of methods.

Numerical fidelity
------------------
Downstream simulation results are validated bit-for-bit, so the *order* of the
floating-point operations below is deliberate: rewriting ``a / b * v`` as
``v * a / b`` changes the last bits of the result. Two behaviours are likewise
preserved on purpose, because they drive branch selection and dtypes:

* ``math.isclose(x, 0.0, rel_tol=...)`` is False for every ``x != 0`` -- a relative
  tolerance against zero degenerates -- so those comparisons only fire on exact
  zeros;
* ``scipy.linalg.sqrtm`` may return a complex matrix. The moment residuals are
  accumulated into a ``float64`` array, which discards the imaginary parts, and the
  simulated sample inherits the complex dtype (callers cast it back to float).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.optimize import root
from scipy.special import gammaln
from scipy.stats import t as student_t

__all__ = ["IG", "fit_pearson_from_moments"]

# Relative tolerance used by the moment-based type selection.
_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Pearson type IV: normalising constant
#
# The type IV density is proportional to  (1 + x^2)^(-m) * exp(-nu * arctan(x)),
# whose normalising constant involves |Gamma(m + i nu/2) / Gamma(m)|^2. The two
# helpers below evaluate that squared modulus (Heinrich, 2004): the recurrence
# raises the argument until the series converges, then the series is summed.
# ---------------------------------------------------------------------------
def _log_gamma_ratio_sq(x: float, y: float) -> float:
    """Return ``log |Gamma(x + i y) / Gamma(x)|^2``."""
    eps = np.finfo(float).eps
    y2 = y * y
    xmin = max(2 * y2, 10.0)
    r = 0.0
    s = 1.0
    p = 1.0
    f = 0.0
    # Recurrence: shift x upwards, accumulating the ratio factors.
    while x < xmin:
        t = y / x
        x += 1.0
        r += np.log(1 + t * t)
    # Series expansion, summed until it falls below machine precision.
    while p > s * eps:
        p *= y2 + f * f
        p /= x * (f + 1)
        f += 1.0
        x += 1.0
        s += p
    return -r - np.log(s)


def _log_type4_norm(m: float, nu: float, a: float) -> float:
    """Log of the Pearson type IV normalising constant."""
    return -np.log(np.sqrt(np.pi)) + _log_gamma_ratio_sq(m, 0.5 * nu) + gammaln(m) - gammaln(m - 0.5) - np.log(a)


# ---------------------------------------------------------------------------
# Pearson type IV: rejection sampler (Heinrich, 2004)
#
# The variate is drawn in the angular variable x = arctan((v - lambda) / a).
# A single uniform draw is folded into a piecewise proposal (uniform core plus
# exponential tails), then accepted against the log-density.
# ---------------------------------------------------------------------------
def _sample_type4_one(m: float, nu: float, a: float, lam: float, log_norm: float) -> float:
    """Draw one Pearson type IV variate, given the log normalising constant."""
    half_pi = np.pi / 2
    b = 2 * m - 2
    mode = np.arctan(-nu / b)
    cos_mode = b / np.sqrt(b * b + nu * nu)
    log_peak = b * np.log(cos_mode) - nu * mode
    width = np.exp(-log_peak - log_norm)

    while True:
        side = 0
        log_tail = 0
        u = 4 * np.random.random()
        # Fold [0, 4) into a side flag and a [0, 2) coordinate.
        if u > 2:
            u -= 2
            side = 1
        # Values above 1 map to the exponential tail of the proposal.
        if u > 1:
            log_tail = np.log(u - 1)
            u = 1 - log_tail
        x = (mode + width * u) if side else (mode - width * u)
        if np.abs(x) < half_pi and log_tail + np.log(np.random.random()) <= b * np.log(np.cos(x)) - nu * x - log_peak:
            break
    return a * np.tan(x) + lam


# ---------------------------------------------------------------------------
# Per-type samplers. Each takes the sample size and the parameter mapping
# produced by :func:`fit_pearson_from_moments`.
# ---------------------------------------------------------------------------
def _sample_type0(n: int, params: dict[str, Any]) -> np.ndarray:
    """Type 0 -- normal distribution."""
    return np.random.normal(params["mean"], params["sd"], n)


def _sample_type1(n: int, params: dict[str, Any]) -> np.ndarray:
    """Type I -- four-parameter beta distribution (bounded support)."""
    return params["scale"] * np.random.beta(params["a"], params["b"], n) + params["location"]


def _sample_type2(n: int, params: dict[str, Any]) -> np.ndarray:
    """Type II -- symmetric special case of type I."""
    a = params["a"]
    return params["scale"] * np.random.beta(a, a, n) + params["location"]


def _sample_type3(n: int, params: dict[str, Any]) -> np.ndarray:
    """Type III -- gamma distribution (possibly reflected)."""
    scale = params["scale"]
    gscale = abs(scale)
    ssgn = np.sign(scale)
    return ssgn * np.random.gamma(params["shape"], gscale, n) + params["location"]


def _sample_type4(n: int, params: dict[str, Any]) -> np.ndarray:
    """Type IV -- skewed, heavy-tailed distribution with unbounded support."""
    m = params["m"]
    nu = params["nu"]
    location = params["location"]
    scale = params["scale"]

    if scale <= 0 or m <= 0.5:
        raise ValueError("Scale must be positive and m must be greater than 0.5")
    if m < 1:
        raise ValueError("Simulation for m < 1 currently not available")

    log_norm = _log_type4_norm(m, nu, 1.0)
    return np.array([_sample_type4_one(m, nu, scale, location, log_norm) for _ in range(n)])


def _sample_type5(n: int, params: dict[str, Any]) -> np.ndarray:
    """Type V -- inverse gamma distribution."""
    scale = params["scale"]
    gscale = 1 / abs(scale)
    ssgn = np.sign(scale)
    return ssgn / np.random.gamma(params["shape"], gscale, n) + params["location"]


def _sample_type6(n: int, params: dict[str, Any]) -> np.ndarray:
    """Type VI -- beta-prime distribution, drawn through a Student's t variate."""
    a = params["a"]
    b = params["b"]
    nscale = params["scale"] * a / b
    return nscale * student_t.rvs(2 * a, loc=0, scale=1, size=n) + params["location"]


def _sample_type7(n: int, params: dict[str, Any]) -> np.ndarray:
    """Type VII -- symmetric Student's t distribution."""
    return params["scale"] * student_t.rvs(df=params["df"], loc=0, scale=1, size=n) + params["location"]


_SAMPLERS = {
    0: _sample_type0,
    1: _sample_type1,
    2: _sample_type2,
    3: _sample_type3,
    4: _sample_type4,
    5: _sample_type5,
    6: _sample_type6,
    7: _sample_type7,
}


def _sample_pearson(n: int, params: dict[str, Any]) -> np.ndarray:
    """Draw ``n`` variates from the fitted Pearson distribution ``params``."""
    pearson_type = params["type"]
    try:
        sampler = _SAMPLERS[pearson_type]
    except KeyError as exc:
        raise ValueError(f"Unknown Pearson type: {pearson_type!r}") from exc
    return sampler(n, params)


# ---------------------------------------------------------------------------
# Moment matching: (mean, variance, skewness, kurtosis) -> Pearson type
# ---------------------------------------------------------------------------
def fit_pearson_from_moments(moments) -> dict[str, Any]:
    """Identify the Pearson distribution matching the first four moments.

    Parameters
    ----------
    moments:
        Sequence ``(mean, variance, skewness, kurtosis)`` -- kurtosis is the raw
        (non-excess) fourth standardised moment.

    Returns
    -------
    dict
        ``{"type": <0..7>, ...}`` with the parameters of the identified law.
    """
    m, v, s, k = moments[0], moments[1], moments[2], moments[3]
    s2 = s**2

    # Boundary of the Pearson region: the moments degenerate to a two-point law.
    if math.isclose(s2, k - 1, rel_tol=_TOLERANCE):
        p = (1 + s / math.sqrt(4 + s2)) / 2
        a = m - v * math.sqrt((1 - p) / p)
        b = m + v * math.sqrt(p / (1 - p))
        raise ValueError(
            f"Target distribution not in Pearson system,\ntry a discrete distribution with mass\n"
            f"p = {p} on a = {a} and mass\n1-p = {1 - p} on b = {b} instead!"
        )
    if s2 > (k - 1):
        raise ValueError("There are no probability distributions with these moments")

    # Coefficients of the Pearson differential equation.
    denom = 10 * k - 12 * s2 - 18
    c0 = (4 * k - 3 * s2) / denom * v
    c1 = s * (k + 3) / denom * math.sqrt(v)
    c2 = (2 * k - 3 * s2 - 6) / denom

    # --- Symmetric branch (exact zero skewness only, see module docstring) ---
    if math.isclose(s, 0, rel_tol=_TOLERANCE):
        if np.isclose(k, 3, _TOLERANCE):
            return {"type": 0, "mean": m, "sd": math.sqrt(v)}

        if k < 3:  # platykurtic -> bounded symmetric beta
            a1 = math.sqrt(v) / 2 * (-math.sqrt(-16 * k * (2 * k - 6)) / (2 * k - 6))
            m1 = -1 / (2 * c2)
            if m1 <= -1:
                raise ValueError("m1 must be greater than -1")
            sca = 2 * a1
            loc = m - sca / 2
            return {"type": 2, "a": 1 + m1, "location": loc, "scale": sca}

        if k > 3:  # leptokurtic -> Student's t
            r = 6 * (k - 1) / (2 * k - 6)
            a = math.sqrt(v * (r - 1))
            dof = 1 + r
            return {"type": 7, "df": dof, "location": m, "scale": a / np.sqrt(dof)}

    # --- Skewed branch, discriminated by kappa ---
    elif not math.isclose(2 * k - 3 * s2 - 6, 0, rel_tol=_TOLERANCE):
        kappa = 0.25 * s2 * (k + 3) ** 2 / ((4 * k - 3 * s2) * (2 * k - 3 * s2 - 6))

        if kappa < 0 or kappa > 1:
            # Roots of the quadratic in the Pearson equation: the two endpoints.
            disc = math.sqrt(s2 * (k + 3) ** 2 - 4 * (4 * k - 3 * s2) * (2 * k - 3 * s2 - 6))
            a1 = math.sqrt(v) / 2 * (-s * (k + 3) - disc) / (2 * k - 3 * s2 - 6)
            a2 = math.sqrt(v) / 2 * (-s * (k + 3) + disc) / (2 * k - 3 * s2 - 6)
            if a1 > 0:
                a1, a2 = a2, a1

            if kappa < 0:  # roots of opposite sign -> bounded beta
                m1 = -(s * (k + 3) + a1 * denom / math.sqrt(v)) / disc
                m2 = -(-s * (k + 3) - a2 * denom / math.sqrt(v)) / disc
                if not (m1 > -1 and m2 > -1):
                    raise ValueError("Both m1 and m2 must be greater than -1")
                sca = a2 - a1
                loc = m - sca * (m1 + 1) / (m1 + m2 + 2)
                return {"type": 1, "a": 1 + m1, "b": 1 + m2, "location": loc, "scale": sca}

            # kappa > 1: roots of the same sign -> beta-prime
            m1 = (c1 + a1) / (c2 * (a2 - a1))
            m2 = -(c1 + a2) / (c2 * (a2 - a1))
            sca = a2 - a1
            loc = m + sca * (m2 + 1) / (m2 + m1 + 2)
            return {"type": 6, "a": 1 + m2, "b": -m2 - m1 - 1, "location": loc, "scale": sca}

        if math.isclose(kappa, 1, rel_tol=_TOLERANCE):  # inverse gamma
            shift = c1 / (2 * c2)
            sca = -(c1 - shift) / c2
            return {"type": 5, "shape": 1 / c2 - 1, "location": m - shift, "scale": sca}

        if 0 < kappa < 1:  # complex roots -> unbounded skewed support
            r = 6 * (k - s2 - 1) / (2 * k - 3 * s2 - 6)
            spread = 16 * (r - 1) - s2 * (r - 2) ** 2
            nu = -r * (r - 2) * s / math.sqrt(spread)
            scale = math.sqrt(v * spread) / 4
            location = m - ((r - 2) * s * math.sqrt(v)) / 4
            return {"type": 4, "m": 1 + r / 2, "nu": nu, "location": location, "scale": scale}

    # --- Degenerate denominator -> gamma ---
    else:
        shape_minus_one = c0 / (c1**2) - 1
        loc = shape_minus_one - c0 / c1
        return {"type": 3, "shape": shape_minus_one + 1, "location": loc, "scale": c1}

    raise ValueError("Unable to identify a Pearson type for these moments")


# ---------------------------------------------------------------------------
# Independent Generator
# ---------------------------------------------------------------------------
def IG(N, sigma_target, skewness, excesskurtosis, reps, typeA):
    """Generate multivariate non-normal data.

    Produces ``reps`` datasets of ``N`` observations whose covariance matrix,
    skewness and excess kurtosis approach the supplied targets. Independent
    Pearson variates are drawn per latent factor, then mixed by a square root of
    the target covariance matrix.

    Parameters
    ----------
    N:
        Number of observations per dataset.
    sigma_target:
        Target covariance matrix (``pandas.DataFrame`` indexed by variable).
    skewness:
        Target skewness of each variable.
    excesskurtosis:
        Target excess kurtosis of each variable.
    reps:
        Number of datasets to simulate.
    typeA:
        ``"triang"`` for a Cholesky factor, otherwise a matrix square root.

    Returns
    -------
    list[pandas.DataFrame]
        ``reps`` DataFrames of simulated data.
    """
    if sigma_target is None:
        raise ValueError("Please specify sigma_target")

    nvar = sigma_target.shape[0]

    # Mixing matrix. Note: sqrtm may return a complex matrix (see module docstring).
    if typeA == "triang":
        A = np.linalg.cholesky(sigma_target).T
    else:
        A = (sqrtm(sigma_target)).T

    def residual_skew(ig_values):
        """Third-moment residual; accumulated in float64 (drops any imaginary part)."""
        fval = np.zeros(nvar)
        for i in range(nvar):
            fval[i] = np.dot(A[i, :] ** 3, ig_values) / (np.sum(A[i, :] ** 2) ** (3 / 2))
        return fval - skewness

    def residual_kurt(ig_values):
        """Fourth-moment residual; accumulated in float64 (drops any imaginary part)."""
        fval = np.zeros(nvar)
        for i in range(nvar):
            fval[i] = np.dot(A[i, :] ** 4, ig_values) / (np.sum(A[i, :] ** 2) ** 2)
        return fval - excesskurtosis

    # Solve for the univariate moments the latent factors must carry.
    ig_skew = root(residual_skew, np.zeros(nvar)).x
    ig_kurt_excess = root(residual_kurt, np.zeros(nvar)).x

    parlist = []
    for i in range(nvar):
        moments = np.array([0, 1, ig_skew[i], 3 + ig_kurt_excess[i]])
        parlist.append(fit_pearson_from_moments(moments))

    ig_data = np.array([_sample_pearson(N * reps, parlist[i]) for i in range(nvar)]).T

    simulated_samples = ig_data.dot(A.T)
    simulated_samples = pd.DataFrame(simulated_samples, columns=sigma_target.columns)

    idx = np.repeat(np.arange(reps), N)
    return [simulated_samples[idx == i] for i in range(reps)]
