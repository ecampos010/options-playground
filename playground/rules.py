"""Small, pure, independently switchable rule functions (Green rules + ladder variants).

Entry blockers return a short reason string ("gate", "upside", ...) or None.
"""
from __future__ import annotations

import math
from datetime import date

from . import bs
from .config import Config

MIN_UNIT_RISK = 150.0   # $ per contract below which no condor can fit (skip fetching quotes)


# ---------------------------------------------------------------- filters
def gate_trips(vix: float, day_chg: float, chg5: float, cfg: Config) -> bool:
    """Hard-gate condition (VIX >= 25 OR day <= -1.5% OR 5d <= -4%), regardless of the switch."""
    return vix >= cfg.gate_vix or day_chg <= cfg.gate_day or chg5 <= cfg.gate_5d


def gate_block(vix, day_chg, chg5, cfg: Config) -> str | None:
    return "gate" if cfg.gate and gate_trips(vix, day_chg, chg5, cfg) else None


def upside_block(chg5: float, cfg: Config) -> str | None:
    return "upside" if cfg.upside_guard and chg5 >= cfg.upside_5d else None


def notrip_block(vix, day_chg, chg5, cfg: Config) -> str | None:
    """Red: enter only when the hard gate trips."""
    if cfg.entry_requires_gate_trip and not gate_trips(vix, day_chg, chg5, cfg):
        return "notrip"
    return None


def event_block(upcoming: list[tuple[str, int]], entry_minute: int, cfg: Config) -> str | None:
    """upcoming = [(event_type, trading-day offset to event)] for the entry day (0 = event day)."""
    if not cfg.event_buffer:
        return None
    for etype, off in upcoming:
        if etype not in cfg.event_types:
            continue
        days = cfg.event_buffer_overrides.get(etype, cfg.event_buffer_days)
        if off > days:
            continue
        if off == 0 and etype in cfg.event_dayof_allowed:
            rel = cfg.event_release_times.get(etype, "00:00")
            h, m = rel.split(":")
            if entry_minute >= int(h) * 60 + int(m):
                continue
        return "event"
    return None


def earnings_block(has_earnings_in_life: bool, cfg: Config) -> str | None:
    return "earnings" if cfg.earnings_filter and has_earnings_in_life else None


def cb_block(day_idx: int, cb_until_idx: int, cfg: Config) -> str | None:
    return "cb" if cfg.cb_days > 0 and day_idx < cb_until_idx else None


def lots_block(n_open: int, cfg: Config) -> str | None:
    return "lots" if cfg.max_open_lots and n_open >= cfg.max_open_lots else None


def cb_trigger(month_start_equity: float, month_end_equity: float) -> bool:
    """A losing calendar month trips the circuit breaker."""
    return month_end_equity < month_start_equity


# ---------------------------------------------------------------- sizing
def contracts(headroom: float, credit: float, cfg: Config) -> int:
    """floor(headroom / ((width - credit) * 100)); --full-use rounds to nearest instead."""
    unit = (cfg.width - credit) * 100.0
    if unit <= 0 or headroom <= 0:
        return 0
    x = headroom / unit
    return int(math.floor(x + 0.5)) if cfg.full_use else int(math.floor(x + 1e-9))


def headrooms(equity: float, open_risk_total: float, open_risk_expiry: float, cfg: Config,
              n_underlyings: int = 1, open_risk_underlying: float = 0.0) -> float:
    budget = cfg.effective_budget() * equity
    h = budget - open_risk_total
    if cfg.expiry_cap:
        h = min(h, cfg.expiry_cap_frac * budget - open_risk_expiry)
    if n_underlyings > 1:
        h = min(h, cfg.underlying_cap_frac * budget - open_risk_underlying)
    return h


# ---------------------------------------------------------------- strikes
def target_strikes(spot: float, vix: float, cfg: Config) -> tuple[float, float]:
    r = cfg.strike_rule
    k = vix / r["vix_ref"]
    return spot * (1 - r["put_pct"] * k), spot * (1 + r["call_pct"] * k)


def nearest_listed(target: float, strikes) -> float:
    return min(strikes, key=lambda s: abs(s - target))


def band(strikes: list[float], center: float, n: int) -> list[float]:
    """Listed strikes within +/- n listed strikes of the one nearest `center`."""
    s = sorted(strikes)
    i = min(range(len(s)), key=lambda j: abs(s[j] - center))
    return s[max(0, i - n): i + n + 1]


def entry_strikes(strikes: list[float], target: float, right: str, cfg: Config) -> tuple[list[float], list[float]]:
    """(band of candidate short strikes, all strikes needed incl. the long wing) for one side."""
    nb = cfg.band_strikes if cfg.strike_rule["mode"] == "vix" else max(cfg.band_strikes, 20)
    bnd = band(strikes, target, nb)
    lo = min(bnd) - cfg.width if right == "P" else min(bnd)
    hi = max(bnd) if right == "P" else max(bnd) + cfg.width
    return bnd, [k for k in strikes if lo <= k <= hi]


def candidate_expiries(expiries: list[date], d: date, cfg: Config) -> list[date]:
    lo, tgt, hi = cfg.dte
    c = [e for e in expiries if lo <= (e - d).days <= hi]
    return sorted(c, key=lambda e: (abs((e - d).days - tgt), e))[: cfg.entry_expiries]


def is_monthly(e: date) -> bool:
    return e.weekday() == 4 and 15 <= e.day <= 21


def rank_expiries(expiries: list[date], d: date, cfg: Config) -> list[date]:
    """All expiries in the DTE window, Fridays (monthlies on ties) first, then closest to target DTE.
    Non-Fridays stay at the end as a fallback; the caller confirms each with a trial quote."""
    lo, tgt, hi = cfg.dte
    c = [e for e in expiries if lo <= (e - d).days <= hi]
    return sorted(c, key=lambda e: (e.weekday() != 4, abs((e - d).days - tgt), not is_monthly(e), e))


def delta_pick(strikes, mids, spot, T, r, right, target_abs_delta) -> float | None:
    """Strike whose implied |delta| (from mid prices) is closest to target."""
    import numpy as np
    k = np.asarray(strikes, float)
    iv = bs.implied_vol(np.asarray(mids, float), spot, k, T, r, right)
    dl = np.abs(bs.delta(spot, k, T, r, iv, right))
    ok = ~np.isnan(dl)
    if not ok.any():
        return None
    idx = np.where(ok)[0][np.argmin(np.abs(dl[ok] - target_abs_delta))]
    return float(k[idx])


# ---------------------------------------------------------------- fills
def leg_entry_price(qty: int, bid: float, ask: float, cfg: Config) -> float:
    """Signed cash per share on entry: + credit (we sell) / - debit (we buy)."""
    mid = 0.5 * (bid + ask)
    if cfg.fill == "natural":
        return bid if qty < 0 else -ask
    return mid if qty < 0 else -mid


def leg_close_cost(qty: int, bid: float, ask: float, fill: str) -> float:
    """Cost per share to close one leg (positive = we pay). Short -> buy at ask; long -> sell at bid."""
    mid = 0.5 * (bid + ask)
    if fill == "natural":
        return ask if qty < 0 else -bid
    return mid if qty < 0 else -mid


def entry_credit(legs, quotes, cfg: Config) -> float:
    """legs=[(qty,...)], quotes=[(bid,ask)] -> net credit per condor (haircut applied if asked)."""
    c = sum(leg_entry_price(l.qty, b, a, cfg) for l, (b, a) in zip(legs, quotes))
    return c - (cfg.haircut if cfg.fill == "haircut" else 0.0)


def close_cost(legs, quotes, cfg: Config, fill: str | None = None) -> float:
    f = fill or cfg.fill
    base = "natural" if f == "natural" else "mid"
    c = sum(leg_close_cost(l.qty, b, a, base) for l, (b, a) in zip(legs, quotes))
    return c + (cfg.haircut if f == "haircut" else 0.0)


def valid_quote(bid: float, ask: float) -> bool:
    return bid == bid and ask == ask and ask > 0 and 0 <= bid <= ask


# ---------------------------------------------------------------- exits
def check_exit(credit: float, nat_cost: float, fill_cost: float, spot: float, short_put: float,
               short_call: float, dte_now: int, entry_dte: int, cfg: Config) -> str | None:
    """Exit trigger at one quote bar. Order: stop > breach > tp > time."""
    if cfg.stop_mult is not None and nat_cost >= cfg.stop_mult * credit:
        return "stop"
    if cfg.breach_exit and (spot <= short_put or spot >= short_call):
        return "breach"
    if fill_cost <= (1.0 - cfg.tp) * credit:
        return "tp"
    if cfg.time_exit > 0 and entry_dte > cfg.time_exit and dte_now <= cfg.time_exit:
        return "time"
    return None


def settle_cost(short_put, long_put, short_call, long_call, spot) -> float:
    """Intrinsic cost to close the condor at expiry settle price."""
    return (max(short_put - spot, 0) - max(long_put - spot, 0)
            + max(spot - short_call, 0) - max(spot - long_call, 0))
