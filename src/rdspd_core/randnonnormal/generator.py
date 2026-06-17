#!/usr/bin/env python3
"""Generateur de variates non normales par le systeme de distributions de Pearson.

Code vendorise depuis le package ``RandNonNormal`` d'origine.
Auteur d'origine : nadahannad (cree le 21 novembre 2023).

Refactoring : les imports ont ete remontes en tete de module (ils etaient repetes
dans chaque fonction). La logique numerique est inchangee afin de preserver
exactement les resultats de simulation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.optimize import root
from scipy.special import gamma, gammaln
from scipy.stats import t as student_t
from scipy.stats import uniform


def pearson_fit_m(moments):
    """Ajuste une distribution du systeme de Pearson a partir de ses 4 premiers moments."""
    m = moments[0]
    v = moments[1]
    s = moments[2]
    k = moments[3]
    # Tolerance pour les comparaisons de flottants.
    tolerance = 1e-9
    if math.isclose(s**2, k - 1, rel_tol=tolerance):
        p = (1 + s / math.sqrt(4 + s**2)) / 2
        a = m - v * math.sqrt((1 - p) / p)
        b = m + v * math.sqrt(p / (1 - p))
        errstr = (
            f"Target distribution not in Pearson system,\ntry a discrete distribution with mass\n"
            f"p = {p} on a = {a} and mass\n1-p = {1 - p} on b = {b} instead!"
        )
        raise ValueError(errstr)

    if (s**2) > (k - 1):
        raise ValueError("There are no probability distributions with these moments")

    c0 = (4 * k - 3 * s**2) / (10 * k - 12 * s**2 - 18) * v
    c1 = s * (k + 3) / (10 * k - 12 * s**2 - 18) * math.sqrt(v)
    c2 = (2 * k - 3 * s**2 - 6) / (10 * k - 12 * s**2 - 18)

    if math.isclose(s, 0, rel_tol=tolerance):
        if np.isclose(k, 3, tolerance):
            return {"type": 0, "mean": m, "sd": math.sqrt(v)}

        if k < 3:
            a1 = math.sqrt(v) / 2 * (-math.sqrt(-16 * k * (2 * k - 6)) / (2 * k - 6))
            m1 = -1 / (2 * c2)
            if m1 <= -1:
                raise ValueError("m1 must be greater than -1")
            sca = 2 * a1
            loc = m - sca / 2
            return {"type": 2, "a": 1 + m1, "location": loc, "scale": sca}

        if k > 3:
            r = 6 * (k - 1) / (2 * k - 6)
            a = math.sqrt(v * (r - 1))
            dof = 1 + r
            return {"type": 7, "df": dof, "location": m, "scale": a / np.sqrt(dof)}

    elif not math.isclose(2 * k - 3 * s**2 - 6, 0, rel_tol=tolerance):
        kap = 0.25 * (s**2) * (k + 3) ** 2 / ((4 * k - 3 * s**2) * (2 * k - 3 * (s**2) - 6))
        if kap < 0:
            a1 = (
                math.sqrt(v)
                / 2
                * (-s * (k + 3) - math.sqrt(s**2 * (k + 3) ** 2 - 4 * (4 * k - 3 * s**2) * (2 * k - 3 * s**2 - 6)))
                / (2 * k - 3 * s**2 - 6)
            )
            a2 = (
                math.sqrt(v)
                / 2
                * (-s * (k + 3) + math.sqrt(s**2 * (k + 3) ** 2 - 4 * (4 * k - 3 * s**2) * (2 * k - 3 * s**2 - 6)))
                / (2 * k - 3 * s**2 - 6)
            )

            if a1 > 0:
                a1, a2 = a2, a1
            a = c1
            m1 = -(
                s * (k + 3) + a1 * (10 * k - 12 * s**2 - 18) / math.sqrt(v)
            ) / math.sqrt(s**2 * (k + 3) ** 2 - 4 * (4 * k - 3 * s**2) * (2 * k - 3 * s**2 - 6))
            m2 = -(
                -s * (k + 3) - a2 * (10 * k - 12 * s**2 - 18) / math.sqrt(v)
            ) / math.sqrt(s**2 * (k + 3) ** 2 - 4 * (4 * k - 3 * s**2) * (2 * k - 3 * s**2 - 6))
            if not (m1 > -1 and m2 > -1):
                raise ValueError("Both m1 and m2 must be greater than -1")
            sca = a2 - a1
            loc = m - sca * (m1 + 1) / (m1 + m2 + 2)
            return {"type": 1, "a": 1 + m1, "b": 1 + m2, "location": loc, "scale": sca}

        if math.isclose(kap, 1, rel_tol=tolerance):
            C1 = c1 / (2 * c2)
            sca = -(c1 - C1) / c2
            return {"type": 5, "shape": 1 / c2 - 1, "location": m - C1, "scale": sca}

        if kap > 1:
            a1 = (
                math.sqrt(v)
                / 2
                * (-s * (k + 3) - math.sqrt(s**2 * (k + 3) ** 2 - 4 * (4 * k - 3 * s**2) * (2 * k - 3 * s**2 - 6)))
                / (2 * k - 3 * s**2 - 6)
            )
            a2 = (
                math.sqrt(v)
                / 2
                * (-s * (k + 3) + math.sqrt(s**2 * (k + 3) ** 2 - 4 * (4 * k - 3 * s**2) * (2 * k - 3 * s**2 - 6)))
                / (2 * k - 3 * s**2 - 6)
            )

            if a1 > 0:
                a1, a2 = a2, a1
            a = c1
            m1 = (a + a1) / (c2 * (a2 - a1))
            m2 = -(a + a2) / (c2 * (a2 - a1))
            sca = a2 - a1
            loc = m + sca * (m2 + 1) / (m2 + m1 + 2)
            return {"type": 6, "a": 1 + m2, "b": -m2 - m1 - 1, "location": loc, "scale": sca}

        if 0 < kap < 1:
            r = 6 * (k - s**2 - 1) / (2 * k - 3 * s**2 - 6)
            nu = -r * (r - 2) * s / math.sqrt(16 * (r - 1) - s**2 * (r - 2) ** 2)
            scale = math.sqrt(v * (16 * (r - 1) - s**2 * (r - 2) ** 2)) / 4
            location = m - ((r - 2) * s * math.sqrt(v)) / 4
            m = 1 + r / 2
            return {"type": 4, "m": m, "nu": nu, "location": location, "scale": scale}

    else:
        m = c0 / (c1**2) - 1
        loc = m - c0 / c1
        sca = c1
        return {"type": 3, "shape": m + 1, "location": loc, "scale": c1}


def rpearson0(n, mean=None, sd=None, params=None):
    """Pearson type 0 : loi normale."""
    mean = params.get("mean")
    sd = params.get("sd")
    return np.random.normal(mean, sd, n)


def rpearsonI(n, params):
    """Pearson type I : loi beta a quatre parametres."""
    a = params.get("a")
    b = params.get("b")
    location = params.get("location")
    scale = params.get("scale")
    return scale * np.random.beta(a, b, n) + location


def rpearsonII(n, params):
    """Pearson type II : cas symetrique de la type I."""
    a = params.get("a")
    location = params.get("location")
    scale = params.get("scale")
    return scale * np.random.beta(a, a, n) + location


def rpearsonIII(n, params):
    """Pearson type III : loi gamma."""
    shape = params.get("shape")
    location = params.get("location")
    scale = params.get("scale")
    gscale = abs(scale)
    ssgn = np.sign(scale)
    return ssgn * np.random.gamma(shape, gscale, n) + location


def gammar2(x, y):
    """Fonction auxiliaire (traduction de la fonction R ``gammar2``)."""
    DBL_EPSILON = np.finfo(float).eps
    y2 = y * y
    xmin = max(2 * y2, 10.0)
    r = 1.0
    s = 1.0
    p = 1.0
    f = 0.0
    while x < xmin:
        t = y / x
        x += 1.0
        r *= 1 + t * t
    while p > s * DBL_EPSILON:
        p *= y2 + f * f
        p /= x * (f + 1)
        f += 1.0
        x += 1.0
        s += p
    return 1.0 / (r * s)


def loggammar2(x, y):
    """Version logarithmique de :func:`gammar2`."""
    DBL_EPSILON = np.finfo(float).eps
    y2 = y * y
    xmin = max(2 * y2, 10.0)
    r = 0.0
    s = 1.0
    p = 1.0
    f = 0.0
    while x < xmin:
        t = y / x
        x += 1.0
        r += np.log(1 + t * t)
    while p > s * DBL_EPSILON:
        p *= y2 + f * f
        p /= x * (f + 1)
        f += 1.0
        x += 1.0
        s += p
    return -r - np.log(s)


def type4norm(m, nu, a):
    """Constante de normalisation de la Pearson type IV."""
    return 0.5 * np.sqrt(2 / np.pi) * gammar2(m, 0.5 * nu) * np.exp(gammaln(m) - gammaln(m - 0.5)) / a


def logtype4norm(m, nu, a):
    """Version logarithmique de :func:`type4norm`."""
    return -np.log(np.sqrt(np.pi)) + loggammar2(m, 0.5 * nu) + gammaln(m) - gammaln(m - 0.5) - np.log(a)


def rpears4(m, nu, a, lam):
    """Tirage d'une variate Pearson type IV (rejet)."""
    M_PI_2 = np.pi / 2
    k = type4norm(m, nu, 1.0)
    b = 2 * m - 2
    M = np.arctan(-nu / b)
    cosM = b / np.sqrt(b * b + nu * nu)
    r = b * np.log(cosM) - nu * M
    rc = np.exp(-r) / k
    while True:
        s = 0
        z = 0
        x = 4 * uniform.rvs()
        if x > 2:
            x -= 2
            s = 1
        if x > 1:
            z = np.log(x - 1)
            x = 1 - z
        x = (M + rc * x) if s else (M - rc * x)
        if np.abs(x) < M_PI_2 and z + np.log(uniform.rvs()) <= b * np.log(np.cos(x)) - nu * x - r:
            break
    return a * np.tan(x) + lam


def rpears4logk(m, nu, a, lam, logk):
    """Variante logarithmique de :func:`rpears4`."""
    M_PI_2 = np.pi / 2
    b = 2 * m - 2
    M = np.arctan(-nu / b)
    cosM = b / np.sqrt(b * b + nu * nu)
    r = b * np.log(cosM) - nu * M
    rc = np.exp(-r - logk)
    while True:
        s = 0
        z = 0
        x = 4 * uniform.rvs()
        if x > 2:
            x -= 2
            s = 1
        if x > 1:
            z = np.log(x - 1)
            x = 1 - z
        x = (M + rc * x) if s else (M - rc * x)
        if np.abs(x) < M_PI_2 and z + np.log(uniform.rvs()) <= b * np.log(np.cos(x)) - nu * x - r:
            break
    return a * np.tan(x) + lam


def rpearsonIV(n, params):
    """Pearson type IV (cas general, asymetrique et leptokurtique)."""
    m = params.get("m")
    nu = params.get("nu")
    location = params.get("location")
    scale = params.get("scale")

    if scale <= 0 or m <= 0.5:
        raise ValueError("Scale must be positive and m must be greater than 0.5")

    if any(isinstance(param, list) for param in [m, nu, location, scale]):
        raise ValueError("Vector-valued parameters are not allowed")

    if m < 1:
        raise ValueError("Simulation for m < 1 currently not available")

    has_gsl = False  # La bibliotheque GSL n'est pas disponible ici.
    logspace = True  # Code en dur dans la fonction R d'origine.

    if logspace:
        if has_gsl:
            k = -0.5 * np.log(np.pi) - gammaln(m - 0.5) + 2 * np.real(np.log(gamma(m + nu / 2 * 1j))) - gammaln(m)
        else:
            k = logtype4norm(m, nu, 1.0)
        return np.array([rpears4logk(m, nu, scale, location, k) for _ in range(n)])
    else:
        if has_gsl:
            k = -0.5 * np.log(np.pi) - gammaln(m - 0.5) + 2 * np.real(np.log(gamma(m + nu / 2 * 1j))) - gammaln(m)
        else:
            k = logtype4norm(m, nu, 1.0)
        return np.array([rpears4(m, nu, scale, location) for _ in range(n)])


def rpearsonV(n, params):
    """Pearson type V : loi gamma inverse."""
    shape = params.get("shape")
    location = params.get("location")
    scale = params.get("scale")
    gscale = 1 / abs(scale)
    ssgn = np.sign(scale)
    return ssgn / np.random.gamma(shape, gscale, n) + location


def rpearsonVI(n, params):
    """Pearson type VI : loi beta-prime (basee sur la loi de Student)."""
    a = params.get("a")
    b = params.get("b")
    location = params.get("location")
    scale = params.get("scale")
    nscale = scale * a / b
    return nscale * student_t.rvs(2 * a, loc=0, scale=1, size=n) + location


def rpearsonVII(n, df, location, scale, params=None):
    """Pearson type VII : loi de Student (symetrique, leptokurtique)."""
    df = params.get("df")
    location = params.get("location")
    scale = params.get("scale")
    return scale * student_t.rvs(df=df, loc=0, scale=1, size=n) + location


def rpearson(n, params):
    """Dispatche vers la fonction de tirage correspondant au type de Pearson."""
    suffix = ["0", "I", "II", "III", "IV", "V", "VI", "VII"]
    fname = f"rpearson{suffix[params['type']]}"
    try:
        pearson_function = globals()[fname]
    except KeyError as exc:
        raise ValueError(f"The function {fname} is not defined.") from exc
    return pearson_function(n=n, params={k: v for k, v in params.items() if k != "type"})


def IG(N, sigma_target, skewness, excesskurtosis, reps, typeA):
    """Independent Generator : genere des donnees multivariees non normales.

    Genere ``reps`` jeux de ``N`` observations dont la matrice de covariance,
    l'asymetrie et l'aplatissement (exces de kurtosis) approchent les cibles fournies.

    Parameters
    ----------
    N:
        Nombre d'observations par jeu de donnees.
    sigma_target:
        Matrice de covariance cible (``pandas.DataFrame`` indexe par variable).
    skewness:
        Asymetrie cible de chaque variable.
    excesskurtosis:
        Exces de kurtosis cible de chaque variable.
    reps:
        Nombre de jeux de donnees a simuler.
    typeA:
        ``"triang"`` pour une factorisation de Cholesky, sinon racine matricielle.

    Returns
    -------
    list[pandas.DataFrame]
        Liste de ``reps`` DataFrames de donnees simulees.
    """
    if sigma_target is None:
        raise ValueError("Please specify sigma_target")

    nvar = sigma_target.shape[0]

    def function_skew(IGvalues):
        fval = np.zeros(nvar)
        for i in range(nvar):
            fval[i] = np.dot(A[i, :] ** 3, IGvalues) / (np.sum(A[i, :] ** 2) ** (3 / 2))
        fval = fval - skewness
        return fval

    def function_kurt(IGvalues):
        fval = np.zeros(nvar)
        for i in range(nvar):
            fval[i] = np.dot(A[i, :] ** 4, IGvalues) / (np.sum(A[i, :] ** 2) ** 2)
        fval = fval - excesskurtosis
        return fval

    if typeA == "triang":
        A = np.linalg.cholesky(sigma_target).T
    else:
        A = (sqrtm(sigma_target)).T

    # Parametres d'asymetrie et d'exces de kurtosis pour chaque variable.
    IG_skew = root(function_skew, np.zeros(nvar)).x
    IG_kurt_excess = root(function_kurt, np.zeros(nvar)).x

    parlist = []
    for i in range(nvar):
        moments = np.array([0, 1, IG_skew[i], 3 + IG_kurt_excess[i]])
        parlist.append(pearson_fit_m(moments))

    IG_data = np.array([rpearson(N * reps, parlist[i]) for i in range(nvar)]).T

    simulated_samples = IG_data.dot(A.T)
    simulated_samples = pd.DataFrame(simulated_samples, columns=sigma_target.columns)

    idx = np.repeat(np.arange(reps), N)
    result = [simulated_samples[idx == i] for i in range(reps)]

    return result
