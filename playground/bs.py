"""Black-Scholes helpers (vectorised). No dividends; r is a flat rate."""
from __future__ import annotations

import math

import numpy as np

try:  # scipy is optional
    from scipy.special import ndtr as _ndtr
except Exception:  # pragma: no cover
    _erf = np.frompyfunc(math.erf, 1, 1)

    def _ndtr(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)).astype(float))


def _d1d2(S, K, T, r, sigma):
    S, K, T, sigma = (np.asarray(v, dtype=float) for v in (S, K, T, sigma))
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-6)
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sq
    return d1, d1 - sq, T


def price(S, K, T, r, sigma, right: str):
    """right is 'C' or 'P'. Returns intrinsic when T is ~0."""
    d1, d2, T = _d1d2(S, K, T, r, sigma)
    S, K = np.asarray(S, float), np.asarray(K, float)
    disc = np.exp(-r * T)
    if right == "C":
        return S * _ndtr(d1) - K * disc * _ndtr(d2)
    return K * disc * _ndtr(-d2) - S * _ndtr(-d1)


def delta(S, K, T, r, sigma, right: str):
    d1, _, _ = _d1d2(S, K, T, r, sigma)
    return _ndtr(d1) if right == "C" else _ndtr(d1) - 1.0


def implied_vol(px, S, K, T, r, right: str, lo=0.01, hi=3.0, iters=45):
    """Vectorised bisection implied vol. Returns nan where price is outside no-arb bounds."""
    px = np.asarray(px, float)
    a = np.full(px.shape, lo)
    b = np.full(px.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        rich = price(S, K, T, r, m, right) > px
        b = np.where(rich, m, b)
        a = np.where(rich, a, m)
    iv = 0.5 * (a + b)
    bad = (px <= price(S, K, T, r, lo, right) + 1e-9) | (px >= price(S, K, T, r, hi, right) - 1e-9)
    return np.where(bad, np.nan, iv)
