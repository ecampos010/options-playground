"""Management / protection building blocks (pure functions; the lab engine applies them).

Triggers: distance (spot within X of a short strike on the tested side) and delta (a short leg's |delta| >= X).
Actions: none | exit | roll_tested | roll_condor | roll_untested | close_tested_spread | hedge_put_call | hedge_shares.
"""
from __future__ import annotations

import math

from . import bs
from .engine import Leg

ACTIONS = ("none", "exit", "roll_tested", "roll_condor", "roll_untested", "close_tested_spread",
           "hedge_put_call", "hedge_shares")


def short_legs(legs: list[Leg]) -> list[Leg]:
    return [l for l in legs if l.right in "PC" and l.qty < 0]


def side_legs(legs: list[Leg], side: str) -> list[Leg]:
    return [l for l in legs if l.right == side]


def tested_side(legs: list[Leg], spot: float) -> tuple[str | None, float]:
    """Side whose short strike is closest to spot, and the relative distance (negative if breached)."""
    best, bd = None, math.inf
    for l in short_legs(legs):
        dist = (spot - l.strike) / spot if l.right == "P" else (l.strike - spot) / spot
        if dist < bd:
            best, bd = l.right, dist
    return best, bd


def distance_hit(legs: list[Leg], spot: float, value: float) -> str | None:
    side, dist = tested_side(legs, spot)
    return side if side is not None and dist <= value else None


def leg_delta(mid: float, spot: float, strike: float, T: float, r: float, right: str):
    return bs.delta_from_mid(mid, spot, strike, T, r, right)


def delta_hit(legs: list[Leg], mids: dict, spot: float, T: float, r: float, value: float) -> str | None:
    """mids: {(right, strike): mid}. Fires on the side of the short leg with the largest |delta| >= value."""
    best, bv = None, 0.0
    for l in short_legs(legs):
        d = leg_delta(mids[(l.right, l.strike)], spot, l.strike, T, r, l.right)
        if d is not None and abs(d) >= value and abs(d) > bv:
            best, bv = l.right, abs(d)
    return best


def net_delta(legs: list[Leg], mids: dict, spot: float, T: float, r: float) -> float | None:
    """Per-contract net delta in shares (options x100 + stock legs x100)."""
    tot = 0.0
    for l in legs:
        if l.right == "S":
            tot += l.qty * 100.0
            continue
        d = leg_delta(mids[(l.right, l.strike)], spot, l.strike, T, r, l.right)
        if d is None:
            return None
        tot += l.qty * d * 100.0
    return tot


def roll_target(side: str, spot: float, strikes: list[float], roll_to_pct: float, old_short: float,
                target_pct: float | None = None) -> float | None:
    """New short strike for a rolled vertical: nearest listed strike to spot*(1 -/+ pct) (pct defaults to
    roll_to_pct). For `roll_tested`/`roll_untested` the new short must end up strictly CLOSER to spot than the old
    short (absolute distance), otherwise there is nothing to roll to; for a condor re-centre (`target_pct` given)
    it only has to differ from the old short."""
    pct = roll_to_pct if target_pct is None else target_pct
    tgt = spot * (1 - pct) if side == "P" else spot * (1 + pct)
    k = min(strikes, key=lambda s: abs(s - tgt))
    if target_pct is not None:
        return k if k != old_short else None
    return k if abs(spot - k) < abs(spot - old_short) else None


def untested(side: str) -> str:
    return "C" if side == "P" else "P"


def wing_for(strikes: list[float], short_k: float, side: str, width: float) -> float | None:
    tgt = short_k - width if side == "P" else short_k + width
    c = [s for s in strikes if (s < short_k if side == "P" else s > short_k)]
    if not c:
        return None
    k = min(c, key=lambda s: abs(s - tgt))
    return k if abs(k - tgt) <= width * 0.51 else None


def hedge_pick(mids: dict, strikes: list[float], spot: float, T: float, r: float, side: str, target: float):
    """Strike on `side` ('P'/'C') whose |delta| (BS from mid) is closest to `target`, only OTM strikes."""
    best, bd = None, 9.0
    for k in strikes:
        if (side == "P" and k >= spot) or (side == "C" and k <= spot) or (side, k) not in mids:
            continue
        d = leg_delta(mids[(side, k)], spot, k, T, r, side)
        if d is not None and abs(abs(d) - target) < bd:
            best, bd = k, abs(abs(d) - target)
    return best
