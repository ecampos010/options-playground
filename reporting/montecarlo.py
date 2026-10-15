"""Block-bootstrap Monte Carlo ruin on replayed calendar-month results.

Resamples whole calendar months (blocks of 1-3 months) WITH replacement. The same month index is applied to every
ticker/sleeve at once (we resample rows of the month matrix), so cross-ticker co-movement is preserved.
Resampling a ~10-year sample understates tails: it can never produce a month worse than the worst one observed.
"""
from __future__ import annotations

import numpy as np

CAVEAT_MC = "Bootstrap of a ~10-year sample understates tails (no month worse than the worst observed can appear)."
_cache: dict = {}


def month_indices(n_months: int, horizon: int, block: int, paths: int, seed: int) -> np.ndarray:
    """(paths, horizon) month indices; each row is consecutive runs of `block` months, start drawn uniformly."""
    block = max(1, min(int(block), max(n_months, 1)))
    key = (n_months, horizon, block, paths, seed)
    if key in _cache:
        return _cache[key]
    rng = np.random.default_rng([seed, n_months, horizon, block])
    nb = -(-horizon // block)
    hi = max(n_months - block + 1, 1)
    starts = rng.integers(0, hi, size=(paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(paths, nb * block)[:, :horizon]
    idx = np.minimum(idx, n_months - 1).astype(np.int32)
    if len(_cache) > 64:
        _cache.clear()
    _cache[key] = idx
    return idx


def simulate(ret: np.ndarray, low: np.ndarray, idx: np.ndarray, start_eq: float, broke_below: float) -> dict:
    """ret/low: per-month return and worst intramonth equity relative to month start (<= ret)."""
    P, H = idx.shape
    R = ret[idx]
    Lw = low[idx]
    growth = np.cumprod(1.0 + R, axis=1)
    E = start_eq * growth
    prev = np.concatenate([np.full((P, 1), start_eq), E[:, :-1]], axis=1)
    lowest = np.clip(prev * (1.0 + Lw), 0.0, None)
    lowest = np.minimum(lowest, E)
    ruined_m = lowest < broke_below
    any_ruin = ruined_m.any(1)
    first = np.where(any_ruin, ruined_m.argmax(1), H)
    # freeze equity at the ruin month (the account stops trading)
    cols = np.arange(H)[None, :]
    frozen = np.where(cols >= first[:, None], np.take_along_axis(E, np.minimum(first, H - 1)[:, None], axis=1), E)
    frozen = np.where(any_ruin[:, None] & (cols >= first[:, None]), np.minimum(frozen, lowest[np.arange(P), np.minimum(first, H - 1)][:, None]), frozen)
    peak_prev = np.maximum.accumulate(np.concatenate([np.full((P, 1), start_eq), E], axis=1), axis=1)[:, :-1]
    dd = 1.0 - lowest / np.maximum(peak_prev, 1e-9)
    maxdd = dd.max(1)
    final = frozen[:, -1]
    ttr = first[any_ruin] + 1
    return {
        "p_broke": float(any_ruin.mean()), "p_dd50": float((maxdd > 0.5).mean()), "p_dd90": float((maxdd > 0.9).mean()),
        "p_below_start": float((final < start_eq).mean()),
        "final_p10": float(np.percentile(final, 10)), "final_p50": float(np.percentile(final, 50)),
        "final_p90": float(np.percentile(final, 90)),
        "median_months_to_ruin": float(np.median(ttr)) if len(ttr) else float("nan"),
    }


def run_mc(ret, low, start_eq, broke_below, horizons=(12, 36), block=2, paths=10000, seed=7) -> dict:
    out = {}
    M = len(ret)
    for h in horizons:
        idx = month_indices(M, h, block, paths, seed)
        r = simulate(ret, low, idx, start_eq, broke_below)
        for k, v in r.items():
            out[f"mc{h}_{k}"] = v
    return out
