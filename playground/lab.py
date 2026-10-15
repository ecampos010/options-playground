"""Lab engine: multi-ticker, multi-strategy, management rules, standardized trade logs (docs/TRADE_LOG_SCHEMA.md).

Reuses the Provider interface, rules.py guards, events and profiles; the original Green engine (engine.py) is
untouched. Differences by design: every trigger/stop/exit is detected on one bar and FILLED on the NEXT available
bar at natural prices; fees are charged per contract per leg; unit mode (`sizing=fixed1`) emits per-contract logs.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import bs, mgmt, rules, schema
from .config import Config, apply_overrides
from .engine import Leg, entry_minutes
from .events import EventIndex
from .provider import CLOSE_MIN, MissingData, Provider
from .strategies import STRATEGIES, Chain, collateral_per_contract, payoff_ps
from .tickers import TickerInfo, spread_width

FAR = 10**9


# ---------------------------------------------------------------- fills (leg-generic, |qty| aware)
def open_credit(legs, quotes, fill: str, haircut: float) -> float:
    c = 0.0
    for l, (b, a) in zip(legs, quotes):
        px = 0.5 * (a + b) if fill != "natural" else (b if l.qty < 0 else a)
        c += -l.qty * px
    return c - (haircut if fill == "haircut" else 0.0)


def close_cost(legs, quotes, fill: str, haircut: float = 0.0) -> float:
    c = 0.0
    for l, (b, a) in zip(legs, quotes):
        px = 0.5 * (a + b) if fill != "natural" else (a if l.qty < 0 else b)
        c += l.qty * -px if l.qty > 0 else -l.qty * px
    return c + (haircut if fill == "haircut" else 0.0)


def intrinsic_cost(legs, S: float) -> float:
    """Cost to close option legs at expiry settlement price S (positive = pay)."""
    return -sum(l.qty * (max(l.strike - S, 0.0) if l.right == "P" else max(S - l.strike, 0.0))
                for l in legs if l.right in "PC")


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7 + 14)


def entry_schedule(days: list[date], i0: int, i1: int, cfg: Config) -> tuple[dict, list]:
    """date -> expiry hint (or None). Monthly/scale-in schedules also return the expiries they target."""
    sched: dict[date, date | None] = {}
    t = cfg.dte[1]
    if cfg.scale_in == 4 or cfg.entry_cadence == "monthly":
        offs = [t + 21, t + 14, t + 7, t] if cfg.scale_in == 4 else [t]
        y, m = days[i0].year, days[i0].month
        end = days[i1] + timedelta(days=t + 60)
        while date(y, m, 1) <= end:
            M = third_friday(y, m)
            j = bisect.bisect_right(days, M) - 1
            Mx = days[j] if (M <= days[-1] and j >= 0) else M       # holiday -> previous trading day
            for off in offs:
                tgt = Mx - timedelta(days=off)
                k = bisect.bisect_left(days, tgt)
                cands = [x for x in (k - 1, k) if 0 <= x < len(days)]
                if not cands:
                    continue
                best = min(cands, key=lambda x: (abs((days[x] - tgt).days), x))
                if i0 <= best <= i1 and abs((days[best] - tgt).days) <= 4:
                    sched[days[best]] = Mx
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return sched, []
    seen_week = {}
    for i in range(i0, i1 + 1):
        d = days[i]
        wk = d.isocalendar()[:2]
        if cfg.entry_cadence == "daily":
            sched[d] = None
        elif cfg.entry_cadence == "weekly":
            if wk not in seen_week:
                seen_week[wk] = True
                sched[d] = None
        elif cfg.entry_cadence == "twice-weekly":
            if (wk, "a") not in seen_week:
                seen_week[(wk, "a")] = True
                sched[d] = None
            elif d.weekday() >= 3 and (wk, "b") not in seen_week:
                seen_week[(wk, "b")] = True
                sched[d] = None
        else:
            raise ValueError(f"unknown entry cadence {cfg.entry_cadence}")
    return sched, []


@dataclass
class LabPos:
    id: int
    strat: str
    expiry: date
    legs: list
    orig_legs: list
    n: int
    entry_day: date
    entry_ts: pd.Timestamp
    entry_bar: int
    entry_dte: int
    entry_spot: float
    entry_vix: float
    term_ratio: float | None
    width: float
    collateral: float
    credit0: float
    cash_ps: float
    fees: float
    spot_source: str
    sp_delta: float | None
    sc_delta: float | None
    weight: float = 1.0
    adj_count: int = 0
    adj_cost_ps: float = 0.0
    hedge_cost_usd: float = 0.0
    hedged: bool = False
    shares: list = field(default_factory=list)      # [(shares per contract, price, is_hedge)]
    q: dict | None = None
    pending: tuple | None = None
    last_nat: float | None = None
    last_mid: float | None = None
    last_spot: float | None = None
    last_mark: float = 0.0
    exdiv: int = 0
    skip_until: date | None = None
    meta_equity: float = 0.0
    entry_dist: dict = field(default_factory=dict)  # side -> short-strike distance from spot at entry (fraction)
    leg_exp: dict = field(default_factory=dict)     # (right, strike) -> expiry for hedge legs on another expiry


@dataclass
class LabResult:
    cfg: Config
    run_id: str
    trades: pd.DataFrame
    marks: pd.DataFrame
    blocked: pd.DataFrame
    meta: dict
    equity: pd.DataFrame | None = None
    sizing: pd.DataFrame | None = None
    config_tag: str = ""


def config_tag(cfg: Config) -> str:
    r = cfg.strike_rule
    parts = [f"d{int(round(r['put_delta'] * 100))}" if r["mode"] == "delta" else "vix"]
    parts.append(f"dte{cfg.dte[1]}")
    tp = cfg.ifly_tp if cfg.strategy == "iron_butterfly" else cfg.tp
    parts.append(f"tp{int(round(tp * 100))}")
    parts.append("sl" + (f"{cfg.stop_mult:g}" if cfg.stop_mult is not None else "none"))
    parts.append(f"t{cfg.time_exit}")
    parts.append({"monthly": "m", "weekly": "w", "twice-weekly": "tw", "daily": "dly"}[cfg.entry_cadence])
    if cfg.trigger and cfg.action != "none":
        parts.append(f"trg{cfg.trigger[0]}{cfg.trigger_value:g}-{cfg.action}")
    if cfg.stop_dollar:
        parts.append(f"sd{cfg.stop_dollar:g}")
    if cfg.scale_in == 4:
        parts.append("scalein4")
    if cfg.sizing == "fixed1":
        parts.append("unit")
    return "_".join(parts)


def guards_on(cfg: Config, unit: bool) -> str:
    g = []
    if cfg.force_entry:
        return "none"
    if cfg.gate:
        g.append("gate")
    if cfg.upside_guard:
        g.append("upside")
    if cfg.event_buffer:
        g.append("event")
    if cfg.cb_days and not unit:
        g.append("cb")
    if cfg.term_filter:
        g.append("term")
    if cfg.post_spike_wait:
        g.append("spike")
    if cfg.min_ivr:
        g.append("ivr")
    if cfg.delta_cap_pct:
        g.append("deltacap")
    return ";".join(g) if g else "none"


def run_id_for(cfg: Config, extra: dict | None = None) -> str:
    d = cfg.to_dict()
    for k in ("start_date", "end_date"):
        d[k] = d.get(k)
    h = hashlib.md5(json.dumps({**d, **(extra or {})}, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{cfg.ticker}_{cfg.strategy}_{cfg.profile}_{h}"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).resolve().parent).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class _Lab:
    def __init__(self, provider: Provider, cfg: Config, events, events_note: str, tinfo: TickerInfo | None,
                 max_entries: int | None, window: dict | None):
        if cfg.dte_target:
            cfg = cfg.replace(dte=[cfg.dte_target - 7, cfg.dte_target, cfg.dte_target + 7])
        self.strat = STRATEGIES[cfg.strategy]
        self.p, self.cfg, self.tinfo, self.window, self.max_entries = provider, cfg, tinfo, window, max_entries
        self.unit = cfg.sizing == "fixed1"
        self.events_note = events_note
        self.daily = provider.get_underlying_daily()
        self.days = list(self.daily.index)
        self.close = self.daily["close"].to_numpy(float)
        self.vix = provider.get_vix_daily().reindex(self.daily.index).ffill().to_numpy(float)
        v3 = provider.get_vix3m_daily()
        self.vix3m = None if v3 is None else v3.reindex(self.daily.index).ffill().to_numpy(float)
        self.evx = EventIndex(events, self.days, cfg.ticker) if events is not None else None
        self.exdiv = sorted(provider.get_exdiv_dates() or [])
        self.exdiv_near: set[date] = set()       # ex-div day and the 2 trading days before it
        self.exdiv_before: set[date] = set()     # the 2 trading days strictly before it
        for x in self.exdiv:
            j = bisect.bisect_left(self.days, x)
            for k in range(max(0, j - 2), min(len(self.days), j + 1)):
                self.exdiv_near.add(self.days[k])
                if k < j:
                    self.exdiv_before.add(self.days[k])
        self.pos: list[LabPos] = []
        self.trades, self.marks, self.blocked, self.sizing = [], [], [], []
        self.cash = cfg.start_equity
        self.cb_until = -1
        self.next_id = 1
        self.delta_src = {"provider": 0, "bs_mid_iv": 0}
        self.spot_days: dict[str, int] = {}
        self.missing_spot_days: list[str] = []
        self.notes: list[str] = []
        self.adjust_log: list[dict] = []
        self.eq_rows = []
        self.n_entries = 0
        self.rid = run_id_for(cfg, {"ticker": cfg.ticker, "events": events_note, "window": window})
        self.breach_on = cfg.breach_exit and self.strat.defaults.get("breach_exit", True)
        self.forced_dte = cfg.forced_close_dte if cfg.forced_close_dte is not None else self.strat.defaults.get("forced_close_dte")
        self.tp = cfg.ifly_tp if cfg.strategy == "iron_butterfly" else cfg.tp
        self.r = cfg.apy

    # ------------------------------------------------------------ helpers
    def _block(self, d: date, reason: str):
        self.blocked.append((str(d), self.cfg.ticker, reason))

    def _T(self, expiry: date, d: date, minute: int) -> float:
        return max(((expiry - d).days * 1440 + (CLOSE_MIN - minute)) / (365.0 * 1440), 1e-6)

    def _equity(self) -> float:
        return self.cash + sum(p.last_mark * p.n for p in self.pos)

    def _fee(self, legs) -> float:
        return self.cfg.fee_per_contract * sum(abs(l.qty) for l in legs if l.right in "PC")

    def _load(self, pos: LabPos, d: date, ts, legs=None):
        try:
            if pos.q is None:
                pos.q = {}
            for l in (legs if legs is not None else pos.legs):
                if l.right == "S" or (l.right, l.strike) in pos.q:
                    continue
                b, a = self.p.leg_arrays(d, pos.leg_exp.get((l.right, l.strike), pos.expiry), l.right, l.strike, ts,
                                         self.cfg.resolution)
                pos.q[(l.right, l.strike)] = (b.tolist(), a.tolist())
        except MissingData:
            pos.q = None

    def _quotes(self, pos: LabPos, b: int):
        out = []
        for l in pos.legs:
            if l.right == "S":
                continue
            bid, ask = pos.q[(l.right, l.strike)]
            bq, aq = bid[b], ask[b]
            if not rules.valid_quote(bq, aq):
                return None
            out.append((bq, aq))
        return out

    def _opt_legs(self, pos):
        return [l for l in pos.legs if l.right != "S"]

    def _share_pnl(self, pos: LabPos, spot: float) -> float:
        return sum(q * (spot - px) for q, px, _ in pos.shares)

    def _slip(self, pos: LabPos, spot: float, hedge_only=False) -> float:
        return sum(abs(q) * spot * self.cfg.stock_slip_bps / 1e4 for q, px, h in pos.shares if h or not hedge_only)

    # ------------------------------------------------------------ main loop
    def run(self) -> LabResult:
        cfg, days = self.cfg, self.days
        i0 = 6
        if cfg.start_date:
            i0 = max(i0, next((j for j, x in enumerate(days) if str(x) >= cfg.start_date), len(days)))
        i1 = len(days) - 1
        if cfg.end_date:
            i1 = max(j for j, x in enumerate(days) if str(x) <= cfg.end_date)
        sched, _ = entry_schedule(days, i0, i1, cfg)
        prev_day, prev_eq, month_eq = None, cfg.start_equity, cfg.start_equity
        for i in range(i0, i1 + 1):
            d = days[i]
            if prev_day is not None and not self.unit:
                self.cash += self.cash * cfg.apy / 365.0 * (d - prev_day).days
                if d.month != prev_day.month:
                    if cfg.cb_days > 0 and rules.cb_trigger(month_eq, prev_eq):
                        self.cb_until = i + cfg.cb_days
                    month_eq = prev_eq
            used = None
            try:
                bars, used = self.p.spot_bars(d, cfg.resolution, cfg.spot_source, cfg.spot_fallback)
                ts, spots = bars.index, bars["spot"].tolist()
                self.spot_days[used] = self.spot_days.get(used, 0) + 1
            except MissingData:
                ts, spots = pd.DatetimeIndex([]), []
                self.missing_spot_days.append(str(d))
            mins = [t.hour * 60 + t.minute for t in ts]
            for pos in self.pos:
                pos.q = None
                if len(ts):
                    pos.q = {}
                    self._load(pos, d, ts)
            entry_bars = []
            if d in sched and (self.max_entries is None or self.n_entries < self.max_entries):
                if not len(ts):
                    self._block(d, "nodata")
                else:
                    for m in entry_minutes(d, cfg):
                        bb = next((k for k, x in enumerate(mins) if x >= m), None)
                        if bb is not None and bb not in entry_bars:
                            entry_bars.append(bb)
            first_reason, entered = None, False
            for b in range(len(ts)):
                for pos in list(self.pos):
                    if pos.q is not None:
                        self._step(pos, i, d, b, ts, spots, mins, used)
                if b in entry_bars and not entered:
                    why = self._try_entry(i, d, b, ts, mins, spots, sched.get(d), used)
                    if why is None:
                        entered = True
                        self.n_entries += 1
                    elif first_reason is None:
                        first_reason = why
            if entry_bars and not entered and first_reason and first_reason != "__logged__":
                self._block(d, first_reason)
            for pos in list(self.pos):
                if pos.expiry <= d:
                    self._settle(pos, i, d)
            # exdiv flags (ITM short call within 2 trading days of ex-div)
            if d in self.exdiv_near:
                for pos in self.pos:
                    if any(l.right == "C" and l.qty < 0 and self.close[i] > l.strike for l in pos.legs):
                        pos.exdiv = 1
            for pos in self.pos:
                self.marks.append((pos.id, str(d), round(pos.last_mark, 4)))
            prev_day, prev_eq = d, self._equity()
            self.eq_rows.append({"date": d, "cash": round(self.cash, 2), "equity": round(prev_eq, 2),
                                 "n_open": len(self.pos)})
        for pos in list(self.pos):               # leave nothing open at the end of data: mark out, reason 'other'
            self._finish(pos, "other", pos.last_nat if pos.last_nat is not None else 0.0,
                         pd.Timestamp(prev_day) + pd.Timedelta(hours=16), prev_day,
                         pos.last_spot if pos.last_spot is not None else pos.entry_spot, settle=False, i_last=i1)
        return self._result(days[i0], days[i1])

    # ------------------------------------------------------------ per-bar position logic
    def _step(self, pos: LabPos, i, d, b, ts, spots, mins, used):
        cfg = self.cfg
        if pos.entry_day == d and b <= pos.entry_bar:
            return
        quotes = self._quotes(pos, b)
        if quotes is None:
            return
        opt = self._opt_legs(pos)
        spot = spots[b]
        nat = close_cost(opt, quotes, "natural")
        pos.last_nat = nat
        pos.last_mid = close_cost(opt, quotes, "mid")
        pos.last_spot = spot
        pos.last_mark = 100.0 * (pos.cash_ps - nat) + self._share_pnl(pos, spot) - pos.fees   # fees so far
        if pos.pending is not None:
            if pos.pending[0] == "exit":
                fc = close_cost(opt, quotes, cfg.fill, cfg.haircut)
                self._finish(pos, pos.pending[1], fc, ts[b], d, spot, settle=False, i_last=i)
            else:
                self._do_action(pos, pos.pending, i, d, b, ts, spots, mins, quotes)
            return
        fillc = close_cost(opt, quotes, cfg.fill, cfg.haircut)
        pnl_nat = pos.cash_ps - nat
        pnl_fill = pos.cash_ps - fillc
        c0 = pos.credit0
        dte_now = (pos.expiry - d).days
        why = None
        loss_usd = -(100.0 * pnl_nat + self._share_pnl(pos, spot))
        if cfg.stop_dollar and loss_usd >= cfg.stop_dollar:
            why = "stop_dollar"
        elif cfg.stop_mult is not None and c0 > 0 and pnl_nat <= -(cfg.stop_mult - 1.0) * c0:
            why = "stop"
        elif self.breach_on and self._breached(pos, spot):
            why = "breach"
        elif c0 > 0 and pnl_fill >= self.tp * c0:
            why = "tp"
        elif self.forced_dte is not None and dte_now <= self.forced_dte:
            why = "time"
        elif cfg.time_exit > 0 and pos.entry_dte > cfg.time_exit and dte_now <= cfg.time_exit:
            why = "time"
        elif cfg.close_before_exdiv and d in self.exdiv_before and self._itm_short_call(pos, spot):
            why = "other"
        if why:
            pos.pending = ("exit", why)
            return
        if cfg.trigger and cfg.action != "none" and pos.skip_until != d:
            side = self._trigger(pos, quotes, spot, d, mins[b])
            if side:
                if cfg.action == "exit":
                    pos.pending = ("exit", "trigger_exit")
                else:
                    pos.pending = ("act", cfg.action, side)

    def _breached(self, pos, spot):
        return any((l.right == "P" and spot <= l.strike) or (l.right == "C" and spot >= l.strike)
                   for l in mgmt.short_legs(pos.legs))

    def _itm_short_call(self, pos, spot):
        return any(l.right == "C" and l.qty < 0 and spot > l.strike for l in pos.legs)

    def _trigger(self, pos, quotes, spot, d, minute):
        cfg = self.cfg
        if cfg.trigger == "distance":
            return mgmt.distance_hit(pos.legs, spot, cfg.trigger_value)
        opt = self._opt_legs(pos)
        mids = {(l.right, l.strike): 0.5 * (b + a) for l, (b, a) in zip(opt, quotes)}
        return mgmt.delta_hit(pos.legs, mids, spot, self._T(pos.expiry, d, minute), self.r, cfg.trigger_value)

    # ------------------------------------------------------------ actions (executed on the bar AFTER the trigger)
    def _do_action(self, pos, pend, i, d, b, ts, spots, mins, quotes):
        cfg = self.cfg
        _, action, side = pend
        pos.pending = None
        spot, minute = spots[b], mins[b]
        opt = self._opt_legs(pos)
        qmap = {(l.right, l.strike): q for l, q in zip(opt, quotes)}
        mids = {k: 0.5 * (v[0] + v[1]) for k, v in qmap.items()}
        skip = lambda: setattr(pos, "skip_until", d)
        if action in ("roll_tested", "roll_condor", "roll_untested", "close_tested_spread"):
            if pos.strat not in ("iron_condor", "iron_butterfly", "bull_put_spread"):
                return skip()
            if action.startswith("roll") and pos.adj_count >= cfg.max_rolls:
                return skip()
            sides = {"roll_tested": [side], "roll_untested": [mgmt.untested(side)],
                     "roll_condor": [side, mgmt.untested(side)], "close_tested_spread": []}[action]
            sides = [s for s in sides if mgmt.side_legs(pos.legs, s)]
            if action == "close_tested_spread":
                legs_out = mgmt.side_legs(pos.legs, side)
                if len(legs_out) == len(pos.legs) or not legs_out:
                    return skip()
                cost = close_cost(legs_out, [qmap[(l.right, l.strike)] for l in legs_out], cfg.fill, cfg.haircut)
                self._apply_adjust(pos, legs_out, [], cost, 0.0, d, ts, "close_tested_spread")
                return
            if not sides:
                return skip()
            try:
                strikes = self.p.list_strikes(d, pos.expiry)
            except MissingData:
                return skip()
            close_legs, new_legs, new_quotes = [], [], []
            for s in sides:
                old = mgmt.side_legs(pos.legs, s)
                old_short = next(l.strike for l in old if l.qty < 0)
                if action == "roll_condor":
                    # re-centre: each short returns to its ORIGINAL distance from the current spot
                    # (both verticals follow the move); sides that would not change are left alone
                    nk = mgmt.roll_target(s, spot, strikes, cfg.roll_to_pct, old_short, pos.entry_dist.get(s, cfg.roll_to_pct))
                    if nk is None:
                        continue
                else:
                    nk = mgmt.roll_target(s, spot, strikes, cfg.roll_to_pct, old_short)
                if nk is None:
                    return skip()
                lk = mgmt.wing_for(strikes, nk, s, pos.width)
                if lk is None:
                    return skip()
                try:
                    q = self.p.get_option_quotes(d, pos.expiry, s, [nk, lk], minute, minute, cfg.resolution)
                except MissingData:
                    return skip()
                q = q[q.ts == ts[b]] if len(q) else q
                qd = {float(k): (float(bq), float(aq)) for k, bq, aq in zip(q.strike, q.bid, q.ask)
                      if rules.valid_quote(bq, aq)}
                if nk not in qd or lk not in qd or qd[nk][0] <= 0:
                    return skip()
                close_legs += old
                nl = [Leg(s, nk, -1), Leg(s, lk, 1)]
                new_legs += nl
                new_quotes += [qd[nk], qd[lk]]
            if not new_legs:
                return skip()
            cost = close_cost(close_legs, [qmap[(l.right, l.strike)] for l in close_legs], cfg.fill, cfg.haircut)
            credit = open_credit(new_legs, new_quotes, cfg.fill, 0.0)
            if cfg.roll_credit_only and credit - cost < 0:
                return skip()
            for l, q in zip(new_legs, new_quotes):
                pos.q = pos.q or {}
            self._apply_adjust(pos, close_legs, new_legs, cost, credit, d, ts, action, new_quotes)
            return
        if action == "hedge_put_call":
            if pos.hedged:
                return
            hs = side
            exp = pos.expiry
            if cfg.hedge_expiry == "next":
                nxt = [e for e in self.p.list_expiries(d) if e > pos.expiry]
                exp = nxt[0] if nxt else pos.expiry
            try:
                snap = self.p.chain_snapshot(d, exp, minute, cfg.resolution, spot)
            except MissingData:
                return skip()
            snap = snap[snap.right == hs]
            hm = {(hs, float(k)): 0.5 * (bq + aq) for k, bq, aq in zip(snap.strike, snap.bid, snap.ask) if aq > 0}
            k = mgmt.hedge_pick(hm, [float(x) for x in snap.strike], spot, self._T(exp, d, minute), self.r, hs,
                                cfg.hedge_delta)
            if k is None:
                return skip()
            row = snap[snap.strike == k].iloc[0]
            if not rules.valid_quote(row.bid, row.ask):
                return skip()
            ask = float(row.ask) if cfg.fill == "natural" else 0.5 * float(row.bid + row.ask)
            pos.cash_ps -= ask
            pos.hedge_cost_usd += ask * 100.0
            pos.fees += cfg.fee_per_contract
            pos.legs = pos.legs + [Leg(hs, k, 1)]
            pos.hedged = True
            try:      # hedge on a different expiry is tracked via its own quote arrays
                b2, a2 = self.p.leg_arrays(d, exp, hs, k, ts, cfg.resolution)
                pos.q[(hs, k)] = (b2.tolist(), a2.tolist())
                if exp != pos.expiry:
                    pos.leg_exp[(hs, k)] = exp
            except MissingData:
                pos.q = None
            self.adjust_log.append({"trade": f"{self.rid}-{pos.id}", "day": str(d), "kind": "hedge_put_call",
                                    "closed": [], "opened": [(hs, k, 1)], "net_credit_ps": round(-ask, 4)})
            return
        if action == "hedge_shares":
            if pos.hedged:
                return
            nd = mgmt.net_delta(pos.legs, mids, spot, self._T(pos.expiry, d, minute), self.r)
            if nd is None or abs(nd) < 1.0:
                return skip()
            q = -round(nd)
            slip = abs(q) * spot * cfg.stock_slip_bps / 1e4
            pos.shares.append((float(q), spot, True))
            self.adjust_log.append({"trade": f"{self.rid}-{pos.id}", "day": str(d), "kind": "hedge_shares",
                                    "closed": [], "opened": [("S", spot, q)], "net_credit_ps": 0.0})
            pos.hedge_cost_usd += slip
            pos.hedged = True
            return
        skip()

    def _apply_adjust(self, pos, close_legs, new_legs, cost, credit, d, ts, kind, new_quotes=None):
        cfg = self.cfg
        self.adjust_log.append({"trade": f"{self.rid}-{pos.id}", "day": str(d), "kind": kind,
                                "closed": [(l.right, l.strike, l.qty) for l in close_legs],
                                "opened": [(l.right, l.strike, l.qty) for l in new_legs],
                                "net_credit_ps": round(credit - cost, 4)})
        pos.cash_ps += credit - cost
        pos.adj_cost_ps += cost - credit
        pos.fees += cfg.fee_per_contract * (sum(abs(l.qty) for l in close_legs) + sum(abs(l.qty) for l in new_legs))
        pos.adj_count += 1
        for l in close_legs:
            pos.legs.remove(l)
            pos.q.pop((l.right, l.strike), None)
        pos.legs.extend(new_legs)
        self._load(pos, d, ts, new_legs)

    # ------------------------------------------------------------ closing
    def _finish(self, pos, reason, cost_ps, ts_exit, d, exit_spot, settle: bool, i_last: int):
        cfg = self.cfg
        exit_fee = 0.0 if settle else self._fee(self._opt_legs(pos))
        slip_exit = sum(abs(q) * exit_spot * cfg.stock_slip_bps / 1e4 for q, px, h in pos.shares if h)
        fees = pos.fees + exit_fee
        hedge_usd = pos.hedge_cost_usd + slip_exit
        pnl = 100.0 * (pos.cash_ps - cost_ps) + self._share_pnl(pos, exit_spot) - fees - slip_exit
        pos.last_mark = pnl
        self.marks.append((pos.id, str(d), round(pnl, 4)))
        self.pos.remove(pos)
        if not self.unit:
            self.cash += pnl * pos.n
        cf = None
        if (pos.adj_count or reason != "expiry") and pos.expiry <= self.days[-1]:
            j = bisect.bisect_left(self.days, pos.expiry)
            if j < len(self.days) and self.days[j] == pos.expiry:
                Sx = float(self.close[j])
                s0 = pos.entry_spot
                cf = 100.0 * (pos.credit0 + payoff_ps(pos.orig_legs, Sx, s0)) - self._fee(pos.orig_legs)
        if reason == "expiry" and not pos.adj_count:
            cf = None
        dist = pd.Timestamp(ts_exit)
        self.trades.append({
            "run_id": self.rid, "trade_id": f"{self.rid}-{pos.id}", "ticker": cfg.ticker, "strategy": pos.strat,
            "level": cfg.level, "profile": cfg.profile, "config_tag": config_tag(cfg),
            "entry_ts": pos.entry_ts.isoformat(), "exit_ts": dist.isoformat(), "expiry": str(pos.expiry),
            "dte_entry": pos.entry_dte, "exit_reason": reason, "width": round(pos.width, 4),
            "collateral_per_contract": round(pos.collateral, 2), "credit_per_contract": round(100.0 * pos.credit0, 2),
            "pnl_per_contract": round(pnl, 2), "fees_per_contract": round(fees, 4), "adj_count": pos.adj_count,
            "adj_cost_per_contract": round(100.0 * pos.adj_cost_ps, 2),
            "hedge_cost_per_contract": round(hedge_usd, 2),
            "cf_pnl_to_expiry_per_contract": None if cf is None else round(cf, 2),
            "entry_spot": round(pos.entry_spot, 4), "exit_spot": round(exit_spot, 4),
            "entry_vix": round(pos.entry_vix, 3), "vix_bucket": schema.vix_bucket(pos.entry_vix),
            "vix_term_ratio": None if pos.term_ratio is None else round(pos.term_ratio, 4),
            "short_put_delta": None if pos.sp_delta is None else round(pos.sp_delta, 4),
            "short_call_delta": None if pos.sc_delta is None else round(pos.sc_delta, 4),
            "exdiv_flag": pos.exdiv, "guards_on": guards_on(cfg, self.unit), "fill_model": cfg.fill,
            "spot_source": pos.spot_source})
        if not self.unit:
            self.sizing.append((f"{self.rid}-{pos.id}", pos.n, round(pos.meta_equity, 2)))

    def _settle(self, pos, i, d):
        S = float(self.close[i])
        near = [l for l in self._opt_legs(pos) if pos.leg_exp.get((l.right, l.strike), pos.expiry) <= pos.expiry]
        cost = intrinsic_cost(near, S)
        for l in self._opt_legs(pos):                 # hedge legs on a later expiry: sold at the last bid
            if l not in near:
                bq = (pos.q or {}).get((l.right, l.strike))
                cost -= l.qty * (bq[0][-1] if bq and bq[0][-1] == bq[0][-1] else 0.0)
        pos.last_spot = S
        self._finish(pos, "expiry", cost, pd.Timestamp(d) + pd.Timedelta(hours=16), d, S, settle=True, i_last=i)

    # ------------------------------------------------------------ entry
    def _try_entry(self, i, d, b, ts, mins, spots, hint, used):
        cfg = self.cfg
        spot, minute = spots[b], mins[b]
        vix = float(self.vix[i - 1])
        day_chg, chg5 = spot / self.close[i - 1] - 1, spot / self.close[i - 5] - 1
        if not cfg.force_entry:
            r = (rules.notrip_block(vix, day_chg, chg5, cfg) or rules.gate_block(vix, day_chg, chg5, cfg)
                 or rules.upside_block(chg5, cfg))
            if not r and self.evx is not None and cfg.event_buffer:
                r = rules.event_block(self.evx.upcoming(d), minute, cfg)
            ratio = None if self.vix3m is None else vix / float(self.vix3m[i - 1])
            if not r and cfg.term_filter and ratio is not None and ratio > cfg.term_ratio_max:
                r = "term"
            if not r and cfg.post_spike_wait > 0:
                for k in range(1, cfg.post_spike_wait + 1):
                    j = i - k
                    if j - 6 >= 0 and self.vix[j - 1] / self.vix[j - 6] - 1 >= cfg.spike_pct:
                        r = "spike"
                        break
            if not r and cfg.min_ivr > 0:
                hist = self.vix[max(0, i - 253): i - 1]
                if len(hist) >= 60 and 100.0 * float((hist < vix).mean()) < cfg.min_ivr:
                    r = "ivr"
            if not r and not self.unit:
                r = rules.cb_block(i, self.cb_until, cfg)
            r = r or rules.lots_block(len(self.pos), cfg)
            if r:
                return r
        else:
            ratio = None if self.vix3m is None else vix / float(self.vix3m[i - 1])
        try:
            exps = [hint] if hint is not None else self.p.entry_expiries(d, cfg, minute, spot, vix)
            if hint is not None:
                lo, _, hi = cfg.dte
                if not (lo <= (hint - d).days <= hi + 7) or not self.p.confirm_expiry(d, hint, cfg, minute, spot, vix):
                    self._block(d, "cadence")
                    return "__logged__"
        except MissingData:
            return "nodata"
        reason = "noquote"
        weight = 0.25 if cfg.scale_in == 4 else 1.0
        eq = self._equity()
        for e in exps:
            try:
                built = self._build(d, e, spot, vix, ts[b], minute)
            except MissingData:
                reason = "nodata"
                continue
            if built is None:
                reason = "noquote"
                continue
            legs, quotes, deltas, width, dsrc = built
            opt = [l for l in legs if l.right != "S"]
            credit = open_credit(opt, quotes, cfg.fill, cfg.haircut)
            coll = collateral_per_contract(legs, credit, spot)
            if self.strat.credit and credit < cfg.min_credit_pct * width - 1e-9:
                reason = "mincredit"
                continue
            if cfg.delta_cap_pct and not cfg.force_entry:
                nd = self._entry_net_delta(legs, quotes, spot, d, e, minute)
                if nd is not None and abs(nd) * spot > cfg.delta_cap_pct / 100.0 * coll:
                    reason = "deltacap"
                    continue
            n = 1
            if not self.unit:
                open_risk = sum(p.collateral * p.n for p in self.pos)
                oe = sum(p.collateral * p.n for p in self.pos if p.expiry == e)
                budget = cfg.effective_budget() * eq * weight
                h = budget - open_risk
                if cfg.expiry_cap:
                    h = min(h, cfg.expiry_cap_frac * budget - oe)
                n = int(math.floor(h / coll + 0.5)) if cfg.full_use else int(math.floor(h / coll + 1e-9))
                if n < 1:
                    reason = "size"
                    continue
            if cfg.strategy == "put_bwb":
                width = legs[0].strike - legs[1].strike          # near-wing distance (documented in the schema)
            sp = next((l.strike for l in legs if l.right == "P" and l.qty < 0), None)
            sc = next((l.strike for l in legs if l.right == "C" and l.qty < 0), None)
            fees = self._fee(opt)
            pos = LabPos(self.next_id, cfg.strategy, e, list(legs), list(legs), n, d, ts[b], b, (e - d).days, spot,
                         vix, ratio, width, coll, credit, credit, fees, used or "provider",
                         None if sp is None else deltas.get(("P", sp)), None if sc is None else deltas.get(("C", sc)),
                         weight)
            pos.meta_equity = eq
            if sp is not None:
                pos.entry_dist["P"] = (spot - sp) / spot
            if sc is not None:
                pos.entry_dist["C"] = (sc - spot) / spot
            if any(l.right == "S" for l in legs):
                pos.shares.append((100.0, spot, False))
            self.next_id += 1
            for k in dsrc:
                self.delta_src[k] += dsrc[k]
            pos.last_mid = close_cost(opt, quotes, "mid")
            pos.last_nat = close_cost(opt, quotes, "natural")
            pos.last_mark = 100.0 * (credit - pos.last_nat) - fees
            pos.q = {}
            self.pos.append(pos)
            self._load(pos, d, ts)
            return None
        return reason

    def _entry_net_delta(self, legs, quotes, spot, d, e, minute):
        opt = [l for l in legs if l.right != "S"]
        mids = {(l.right, l.strike): 0.5 * (b + a) for l, (b, a) in zip(opt, quotes)}
        return mgmt.net_delta(legs, mids, spot, self._T(e, d, minute), self.r)

    def _build(self, d, e, spot, vix, ts_b, minute):
        cfg = self.cfg
        strikes = self.p.list_strikes(d, e)
        width = spread_width(spot, strikes, cfg, self.tinfo)
        qd, dsrc, deltas = {}, {"provider": 0, "bs_mid_iv": 0}, {}
        T = self._T(e, d, minute)
        if cfg.strike_rule["mode"] == "delta":
            snap = self.p.chain_snapshot(d, e, minute, cfg.resolution, spot)
            has_delta = "delta" in snap.columns
            for row in snap.itertuples():
                if not rules.valid_quote(row.bid, row.ask):
                    continue
                k = round(float(row.strike), 4)
                qd[(row.right, k)] = (float(row.bid), float(row.ask))
                dl = getattr(row, "delta", None) if has_delta else None
                if dl is None or dl != dl:
                    dl = bs.delta_from_mid(0.5 * (row.bid + row.ask), spot, k, T, self.r, row.right)
                    if dl is not None:
                        dsrc["bs_mid_iv"] += 1
                else:
                    dsrc["provider"] += 1
                if dl is not None:
                    deltas[(row.right, k)] = float(dl)
        else:
            need = self.strat.needed_strikes(strikes, cfg, spot, vix, width)
            for right, ks in need.items():
                df = self.p.get_option_quotes(d, e, right, ks, minute, minute, cfg.resolution)
                df = df[df.ts == ts_b] if len(df) else df
                for k, bq, aq in zip(df.strike, df.bid, df.ask):
                    if rules.valid_quote(bq, aq):
                        qd[(right, round(float(k), 4))] = (float(bq), float(aq))
        ch = Chain(cfg, spot, vix, strikes, qd, deltas, width)
        legs = self.strat.build(ch)
        if legs is None:
            return None
        quotes = [ch.quote(l.right, l.strike) for l in legs if l.right != "S"]
        # entry delta log for shorts (BS from mid when the chain had no delta, e.g. vix-distance mode)
        for l in legs:
            if l.right in "PC" and l.qty < 0 and (l.right, round(l.strike, 4)) not in deltas:
                dl = bs.delta_from_mid(0.5 * sum(ch.quote(l.right, l.strike)), spot, l.strike, T, self.r, l.right)
                if dl is not None:
                    deltas[(l.right, l.strike)] = dl
                    dsrc["bs_mid_iv"] += 1
        deltas = {(r, k): v for (r, k), v in deltas.items()}
        return legs, quotes, deltas, width, dsrc

    # ------------------------------------------------------------ result
    def _result(self, d0, d1) -> LabResult:
        cfg = self.cfg
        t = pd.DataFrame(self.trades, columns=schema.TRADE_COLUMNS)
        m = pd.DataFrame([(f"{self.rid}-{a}", b, c) for a, b, c in self.marks], columns=schema.MARK_COLUMNS)
        bl = pd.DataFrame(self.blocked, columns=schema.BLOCKED_COLUMNS)
        er = dict(t["exit_reason"].value_counts()) if len(t) else {}
        meta = {
            "counts": {"trades": int(len(t)), "exit_reasons": {k: int(v) for k, v in er.items()},
                       "blocked": {k: int(v) for k, v in dict(bl["reason"].value_counts()).items()}},
            "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                         "playground": __import__("playground").__version__},
            "spot_coverage": {"days_by_source": self.spot_days, "days_without_spot": len(self.missing_spot_days),
                              "first_missing": self.missing_spot_days[:10]},
            "delta_source": self.delta_src, "unit_mode": self.unit, "notes": self.notes,
            "events": self.events_note, "fill_model": cfg.fill, "fee_per_contract": cfg.fee_per_contract,
            "config_tag": config_tag(cfg), "level": cfg.level, "window": self.window,
            "adjustments": self.adjust_log}
        eq = pd.DataFrame(self.eq_rows) if not self.unit else None
        sz = pd.DataFrame(self.sizing, columns=["trade_id", "contracts", "equity_at_entry"]) if not self.unit else None
        meta["data_range"] = [str(d0), str(d1)]
        return LabResult(cfg, self.rid, t, m, bl, meta, eq, sz, config_tag(cfg))


def run_lab(provider: Provider, cfg: Config, events=None, events_note: str = "", tinfo: TickerInfo | None = None,
            max_entries: int | None = None, window: dict | None = None) -> LabResult:
    if tinfo is None:
        from .tickers import load_tickers
        tinfo = load_tickers().get(cfg.ticker)
    if cfg.strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {cfg.strategy}; choose from {sorted(STRATEGIES)}")
    if cfg.action not in mgmt.ACTIONS:
        raise ValueError(f"unknown action {cfg.action}")
    if cfg.event_buffer and events is None:
        raise ValueError("event_buffer is on but no events were supplied")
    return _Lab(provider, cfg, events, events_note, tinfo, max_entries, window).run()


def save_lab_run(res: LabResult, runs_dir: str | Path = "runs", command: str = "") -> Path:
    out = Path(runs_dir) / res.run_id
    out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / "trades.csv", index=False)
    res.marks.to_csv(out / "marks.csv", index=False)
    res.blocked.to_csv(out / "blocked_entries.csv", index=False)
    if res.equity is not None:
        res.equity.to_csv(out / "equity.csv", index=False)
        res.sizing.to_csv(out / "sizing.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "flags": res.cfg.to_dict(), "git_commit": git_commit(), "data_range": res.meta["data_range"],
        "window": res.meta["window"], "command": command}, indent=2, default=str))
    (out / "meta.json").write_text(json.dumps(res.meta, indent=2, default=str))
    return out
