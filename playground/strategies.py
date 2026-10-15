"""Defined-risk strategies: build_legs, exact max loss, entry checks, exit-rule defaults.

A structure is a list of signed Legs per ONE contract: right 'P'/'C' (qty -1 short, +1 long, +-2 for the BWB body)
or right 'S' for 100 shares (qty +1 = 100 shares; used by covered_call). Undefined-risk strategies (short straddle /
strangle, ratio spreads, naked jade lizard) are deliberately NOT implemented; max_loss_ps raises for them.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from . import rules
from .config import Config
from .engine import Leg


# ---------------------------------------------------------------- payoff math
def payoff_ps(legs: list[Leg], S: float, S0: float = 0.0) -> float:
    """Expiry payoff per share-of-option-unit (before credit): options intrinsic, shares (S-S0)."""
    tot = 0.0
    for l in legs:
        if l.right == "P":
            tot += l.qty * max(l.strike - S, 0.0)
        elif l.right == "C":
            tot += l.qty * max(S - l.strike, 0.0)
        else:
            tot += l.qty * (S - S0)
    return tot


def max_loss_ps(legs: list[Leg], credit_ps: float, S0: float = 0.0) -> float:
    """Exact worst case (positive number, per share) of credit + expiry payoff. Raises if unbounded."""
    strikes = sorted({l.strike for l in legs if l.right in "PC"})
    hi = max(strikes + [S0, 1.0]) * 10
    slope_up = sum(l.qty for l in legs if l.right == "C") + sum(l.qty for l in legs if l.right == "S")
    if slope_up < 0:
        raise ValueError("undefined risk (net short calls) - not supported")
    pts = [0.0] + strikes + [S0] + [hi]
    worst = min(credit_ps + payoff_ps(legs, s, S0) for s in pts)
    return max(0.0, -worst)


def collateral_per_contract(legs: list[Leg], credit_ps: float, S0: float) -> float:
    return round(max_loss_ps(legs, credit_ps, S0) * 100.0, 4)


# ---------------------------------------------------------------- chain view
class Chain:
    """Quotes + deltas at the entry bar for strike selection."""

    def __init__(self, cfg: Config, spot: float, vix: float, strikes: list[float], quotes: dict, deltas: dict,
                 width: float):
        self.cfg, self.spot, self.vix, self.width = cfg, spot, vix, width
        self.strikes = sorted(strikes)
        self.q, self.d = quotes, deltas
        self._ks = {r: sorted(k for (rr, k) in quotes if rr == r) for r in "PC"}

    def has(self, right: str, k: float) -> bool:
        return (right, round(k, 4)) in self.q

    def quote(self, right: str, k: float):
        return self.q[(right, round(k, 4))]

    def delta(self, right: str, k: float):
        return self.d.get((right, round(k, 4)))

    def listed(self, k: float) -> float | None:
        """Nearest LISTED strike with a valid quote on either right within one width, else None."""
        c = sorted(set(self._ks["P"]) | set(self._ks["C"]))
        if not c:
            return None
        s = min(c, key=lambda x: abs(x - k))
        return s if abs(s - k) <= self.width else None

    def wing(self, right: str, short_k: float, away: int, dist: float | None = None) -> float | None:
        """Quoted strike nearest to `dist` (default width) beyond short_k (away=+1 above / -1 below)."""
        tgt = short_k + away * (self.width if dist is None else dist)
        ks = self._ks[right]
        i = bisect.bisect_left(ks, tgt)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ks) and (ks[j] - short_k) * away > 1e-9:
                if best is None or abs(ks[j] - tgt) < abs(best - tgt):
                    best = ks[j]
        if best is None or abs(best - tgt) > max(self.width * 0.51, 1e-9):
            return None
        return best

    def short(self, right: str, target_delta: float | None = None) -> float | None:
        """Short strike by the configured rule (vix-distance or delta). Needs bid>0 and a long wing."""
        away = -1 if right == "P" else 1
        rule = self.cfg.strike_rule
        ks = [s for s in self._ks[right] if self.q[(right, s)][0] > 0 and self.wing(right, s, away) is not None]
        if not ks:
            return None
        if rule["mode"] == "delta":
            tgt = target_delta if target_delta is not None else rule["put_delta" if right == "P" else "call_delta"]
            ok = [s for s in ks if self.delta(right, s) is not None]
            return min(ok, key=lambda s: abs(abs(self.delta(right, s)) - tgt)) if ok else None
        tp, tc = rules.target_strikes(self.spot, self.vix, self.cfg)
        tgt = tp if right == "P" else tc
        return min(ks, key=lambda s: abs(s - tgt))


@dataclass
class Strategy:
    name: str = ""
    defaults: dict = field(default_factory=dict)      # cfg fields applied unless explicitly overridden
    credit: bool = True                               # min-credit rule applies (10% of width)
    sides: str = "PC"
    experimental: bool = False

    def needed_strikes(self, strikes: list[float], cfg: Config, spot: float, vix: float, width: float) -> dict:
        """{right: [strikes to quote]} for the tight (vix-distance) fetch."""
        tp, tc = rules.target_strikes(spot, vix, cfg)
        out = {}
        for r in self.sides:
            tgt = tp if r == "P" else tc
            _, need = rules.entry_strikes(strikes, tgt, r, cfg, width)
            out[r] = need
        return out

    def build(self, ch: Chain) -> list[Leg] | None:
        raise NotImplementedError

    def entry_check(self, ctx: dict) -> str | None:
        return None


class IronCondor(Strategy):
    def __init__(self):
        super().__init__("iron_condor", {"breach_exit": True})

    def build(self, ch):
        sp, sc = ch.short("P"), ch.short("C")
        if sp is None or sc is None or sp >= sc:
            return None
        lp, lc = ch.wing("P", sp, -1), ch.wing("C", sc, +1)
        if lp is None or lc is None:
            return None
        return [Leg("P", sp, -1), Leg("P", lp, 1), Leg("C", sc, -1), Leg("C", lc, 1)]


class BullPutSpread(Strategy):
    def __init__(self):
        super().__init__("bull_put_spread", {"breach_exit": True}, sides="P")

    def build(self, ch):
        sp = ch.short("P")
        lp = None if sp is None else ch.wing("P", sp, -1)
        return None if lp is None else [Leg("P", sp, -1), Leg("P", lp, 1)]


class IronButterfly(Strategy):
    def __init__(self):
        super().__init__("iron_butterfly", {"breach_exit": False, "forced_close_dte": 2})

    def needed_strikes(self, strikes, cfg, spot, vix, width):
        out = {}
        for r in "PC":
            _, need = rules.entry_strikes(strikes, spot, r, cfg, width)
            out[r] = need
        return out

    def build(self, ch):
        both = [s for s in ch.strikes if ch.has("P", s) and ch.has("C", s)
                and ch.quote("P", s)[0] > 0 and ch.quote("C", s)[0] > 0]
        if not both:
            return None
        k = min(both, key=lambda s: abs(s - ch.spot))
        lp, lc = ch.wing("P", k, -1), ch.wing("C", k, +1)
        if lp is None or lc is None:
            return None
        return [Leg("P", k, -1), Leg("P", lp, 1), Leg("C", k, -1), Leg("C", lc, 1)]


class PutBWB(Strategy):
    """Buy 1 near-wing put (K1) / sell 2 body puts (K2) / buy 1 far-wing put (K3 < K2). Max loss exact via payoff."""

    def __init__(self):
        super().__init__("put_bwb", {"breach_exit": False}, credit=False, sides="P")

    def needed_strikes(self, strikes, cfg, spot, vix, width):
        tp, _ = rules.target_strikes(spot, vix, cfg)
        _, near = rules.entry_strikes(strikes, tp, "P", cfg, width)
        far_t = tp - cfg.bwb_far_pct * spot
        _, far = rules.entry_strikes(strikes, far_t, "P", cfg, width)
        up_t = tp + cfg.bwb_near_pct * spot
        _, up = rules.entry_strikes(strikes, up_t, "P", cfg, width)
        return {"P": sorted(set(near) | set(far) | set(up))}

    def build(self, ch):
        body = ch.short("P")
        if body is None:
            return None
        cfg, spot = ch.cfg, ch.spot
        near_w, far_w = cfg.bwb_near_pct * spot, cfg.bwb_far_pct * spot
        up = [s for s in ch.strikes if ch.has("P", s) and s > body]
        dn = [s for s in ch.strikes if ch.has("P", s) and s < body]
        if not up or not dn:
            return None
        k1 = min(up, key=lambda s: abs(s - (body + near_w)))
        k3 = min(dn, key=lambda s: abs(s - (body - far_w)))
        if not (k3 < body < k1):
            return None
        return [Leg("P", k1, 1), Leg("P", body, -2), Leg("P", k3, 1)]


class CashSecuredPut(Strategy):
    def __init__(self):
        super().__init__("cash_secured_put", {"breach_exit": False}, credit=False, sides="P", experimental=True)

    def build(self, ch):
        # short put by rule; no long wing is required
        away_ok = [s for s in ch.strikes if ch.has("P", s) and ch.quote("P", s)[0] > 0]
        if not away_ok:
            return None
        rule = ch.cfg.strike_rule
        if rule["mode"] == "delta":
            ok = [s for s in away_ok if ch.delta("P", s) is not None]
            if not ok:
                return None
            sp = min(ok, key=lambda s: abs(abs(ch.delta("P", s)) - rule["put_delta"]))
        else:
            tp, _ = rules.target_strikes(ch.spot, ch.vix, ch.cfg)
            sp = min(away_ok, key=lambda s: abs(s - tp))
        return [Leg("P", sp, -1)]


class CoveredCall(Strategy):
    def __init__(self):
        super().__init__("covered_call", {"breach_exit": False}, credit=False, sides="C", experimental=True)

    def build(self, ch):
        ok = [s for s in ch.strikes if ch.has("C", s) and ch.quote("C", s)[0] > 0 and s >= ch.spot]
        if not ok:
            return None
        rule = ch.cfg.strike_rule
        if rule["mode"] == "delta":
            okd = [s for s in ok if ch.delta("C", s) is not None]
            if not okd:
                return None
            sc = min(okd, key=lambda s: abs(abs(ch.delta("C", s)) - rule["call_delta"]))
        else:
            _, tc = rules.target_strikes(ch.spot, ch.vix, ch.cfg)
            sc = min(ok, key=lambda s: abs(s - tc))
        return [Leg("S", 0.0, 1), Leg("C", sc, -1)]


STRATEGIES = {s.name: s for s in (IronCondor(), IronButterfly(), BullPutSpread(), PutBWB(), CashSecuredPut(), CoveredCall())}
