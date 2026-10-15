"""Account/position loop. Pure function of (provider, config, events); deterministic given cfg.seed.

Extension point for hedges (see reserve.py): a Position is a list of signed Legs plus `kind`;
cash, marking, closing and equity code are leg-generic. Only exit *rules* are condor-specific
(dispatched on pos.kind).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from . import rules
from .config import Config
from .events import EventIndex
from .provider import MissingData, Provider, bar_minutes

REASONS = ("tp", "stop", "breach", "time", "expiry", "end")


@dataclass
class Leg:
    right: str      # 'P' / 'C'
    strike: float
    qty: int        # -1 short, +1 long


@dataclass
class Position:
    id: int
    kind: str
    expiry: date
    legs: list
    n: int
    credit: float
    entry_day: date
    entry_ts: pd.Timestamp
    entry_bar: int
    entry_dte: int
    risk: float
    last_mid: float
    meta: dict = field(default_factory=dict)
    q: list | None = None       # per-leg (bid list, ask list) for the current day

    @property
    def sp(self): return self.legs[0].strike
    @property
    def lp(self): return self.legs[1].strike
    @property
    def sc(self): return self.legs[2].strike
    @property
    def lc(self): return self.legs[3].strike


@dataclass
class Result:
    cfg: Config
    provider: str
    trades: pd.DataFrame
    equity: pd.DataFrame
    blocked: dict
    gaps: int
    events_note: str


def entry_minutes(d: date, cfg: Config) -> list[int]:
    """Target entry minutes for day d (deterministic in seed and date, independent of account state)."""
    grid = bar_minutes(cfg.resolution)
    if cfg.resolution == "eod":
        return grid
    w0, w1 = cfg.window()
    inwin = [m for m in grid if w0 <= m <= w1]
    if not inwin:
        return []
    if cfg.entry_sample == "first":
        return [inwin[0]]
    if cfg.entry_sample == "every-30m":
        return [m for m in inwin if (m - inwin[0]) % 30 == 0]
    rng = random.Random(cfg.seed * 1_000_003 + d.toordinal())
    return [rng.choice(inwin)]


def _bar_for(mins: list[int], target: int) -> int | None:
    for i, m in enumerate(mins):
        if m >= target:
            return i
    return None


class _Run:
    def __init__(self, provider: Provider, cfg: Config, events: pd.DataFrame | None, ev_note: str):
        self.p, self.cfg, self.ev_note = provider, cfg, ev_note
        self.daily = provider.get_underlying_daily()
        self.days = list(self.daily.index)
        self.close = self.daily["close"].to_numpy(float)
        self.vix = provider.get_vix_daily().reindex(self.daily.index).ffill().to_numpy(float)
        self.evx = EventIndex(events, self.days) if events is not None else None
        self.cash = cfg.start_equity
        self.pos: list[Position] = []
        self.trades: list[dict] = []
        self.eq_rows: list[dict] = []
        self.blocked: dict = {}
        self.gaps = 0
        self.next_id = 1
        self.cb_until = -1

    # ------------------------------------------------------------ helpers
    def equity(self) -> float:
        return self.cash - sum(p.last_mid * 100 * p.n for p in self.pos)

    def _load_day(self, pos: Position, d: date, ts):
        try:
            pos.q = [self.p.leg_arrays(d, pos.expiry, l.right, l.strike, ts, self.cfg.resolution)
                     for l in pos.legs]
            pos.q = [(b.tolist(), a.tolist()) for b, a in pos.q]
        except MissingData:
            pos.q = None
            self.gaps += 1

    def _close(self, pos: Position, reason: str, cost: float, ts, d: date):
        cfg = self.cfg
        comm = 0.0 if reason in ("expiry",) else cfg.commission * len(pos.legs) * pos.n
        self.cash -= cost * 100 * pos.n + comm
        pnl = (pos.credit - cost) * 100 * pos.n - comm - pos.meta.get("entry_comm", 0.0)
        self.pos.remove(pos)
        self.trades.append({
            "id": pos.id, "kind": pos.kind, "entry_ts": pos.entry_ts, "exit_ts": ts,
            "expiry": pos.expiry, "entry_dte": pos.entry_dte, "short_put": pos.sp, "long_put": pos.lp,
            "short_call": pos.sc, "long_call": pos.lc, "contracts": pos.n, "credit": round(pos.credit, 4),
            "exit_cost": round(cost, 4), "pnl": round(pnl, 2), "exit_reason": reason,
            "hold_days": (d - pos.entry_day).days, "risk": round(pos.risk, 2), "fill": cfg.fill,
            **{k: v for k, v in pos.meta.items() if k != "entry_comm"}})

    def _block(self, reason: str):
        self.blocked[reason] = self.blocked.get(reason, 0) + 1

    # ------------------------------------------------------------ entry
    def _try_entry(self, i: int, d: date, b: int, ts, mins, spots) -> str | None:
        cfg = self.cfg
        spot = spots[b]
        vix = self.vix[i - 1]
        day_chg = spot / self.close[i - 1] - 1
        chg5 = spot / self.close[i - 5] - 1
        r = (rules.notrip_block(vix, day_chg, chg5, cfg) or rules.gate_block(vix, day_chg, chg5, cfg)
             or rules.upside_block(chg5, cfg))
        if not r and self.evx is not None:
            r = rules.event_block(self.evx.upcoming(d), mins[b], cfg)
        r = r or rules.cb_block(i, self.cb_until, cfg) or rules.lots_block(len(self.pos), cfg)
        if r:
            return r
        eq = self.equity()
        budget = cfg.effective_budget() * eq
        open_total = sum(p.risk for p in self.pos)
        if budget - open_total < rules.MIN_UNIT_RISK:
            return "size"
        try:
            exps = self.p.entry_expiries(d, cfg, mins[b], spot, vix)
        except MissingData:
            return "nodata"
        reason = "noexpiry"
        for e in exps:
            if self.evx is not None and self.evx.earnings and rules.earnings_block(
                    self.evx.earnings_between(d, e), cfg):
                reason = "earnings"
                continue
            open_exp = sum(p.risk for p in self.pos if p.expiry == e)
            h = rules.headrooms(eq, open_total, open_exp, cfg)
            if h < rules.MIN_UNIT_RISK:
                reason = "size"
                continue
            try:
                built = self._build_condor(d, e, spot, vix, ts[b], mins[b])
            except MissingData:
                reason = "nodata"
                continue
            if built is None:
                reason = "noquote"
                continue
            legs, quotes = built
            credit = rules.entry_credit(legs, quotes, cfg)
            if credit < cfg.min_credit:
                reason = "credit"
                continue
            n = rules.contracts(h, credit, cfg)
            if n < 1:
                reason = "size"
                continue
            comm = cfg.commission * 4 * n
            risk = (cfg.width - credit) * 100 * n
            pos = Position(self.next_id, "condor", e, legs, n, credit, d, ts[b], b, (e - d).days, risk,
                           rules.close_cost(legs, quotes, cfg, "mid"),
                           {"spot": round(spot, 2), "vix": round(float(vix), 2), "equity": round(eq, 2),
                            "budget": round(budget, 2), "entry_comm": comm})
            self.next_id += 1
            self.cash += credit * 100 * n - comm
            self.pos.append(pos)
            pos.meta["risk_total_after"] = round(sum(p.risk for p in self.pos), 2)
            pos.meta["risk_expiry_after"] = round(sum(p.risk for p in self.pos if p.expiry == e), 2)
            self._load_day(pos, d, ts)
            return None
        return reason

    def _build_condor(self, d, e, spot, vix, ts_b, minute):
        cfg = self.cfg
        strikes = self.p.list_strikes(d, e)
        tp_, tc_ = rules.target_strikes(spot, vix, cfg)
        sides = {}
        for right, tgt in (("P", tp_), ("C", tc_)):
            bnd, need = rules.entry_strikes(strikes, tgt, right, cfg)
            df = self.p.get_option_quotes(d, e, right, need, minute, minute, cfg.resolution)
            df = df[df.ts == ts_b] if len(df) else df
            qd = {float(k): (float(bq), float(aq)) for k, bq, aq in zip(df.strike, df.bid, df.ask)
                  if rules.valid_quote(bq, aq)}
            if cfg.strike_rule["mode"] == "delta":
                T = ((e - d).days + 0.0) / 365.0
                key = "put_delta" if right == "P" else "call_delta"
                ks = [k for k in bnd if k in qd and qd[k][0] > 0
                      and (k - cfg.width if right == "P" else k + cfg.width) in qd]
                if not ks:
                    return None
                best = rules.delta_pick(ks, [0.5 * sum(qd[k]) for k in ks], spot, T, 0.04, right,
                                        cfg.strike_rule[key])
                order = [] if best is None else [best]
            else:
                order = sorted([k for k in bnd], key=lambda k: abs(k - tgt))
            pick = None
            for k in order:
                lng = k - cfg.width if right == "P" else k + cfg.width
                if k in qd and qd[k][0] > 0 and lng in qd:
                    pick = (k, lng, qd)
                    break
            if pick is None:
                return None
            sides[right] = pick
        (sp, lp, qp), (sc, lc, qc) = sides["P"], sides["C"]
        legs = [Leg("P", sp, -1), Leg("P", lp, 1), Leg("C", sc, -1), Leg("C", lc, 1)]
        quotes = [qp[sp], qp[lp], qc[sc], qc[lc]]
        return legs, quotes

    # ------------------------------------------------------------ main
    def run(self) -> Result:
        cfg = self.cfg
        days = self.days
        i0 = 6
        if cfg.start_date:
            i0 = max(i0, next((j for j, x in enumerate(days) if str(x) >= cfg.start_date), len(days)))
        i1 = len(days) - 1
        if cfg.end_date:
            i1 = max(j for j, x in enumerate(days) if str(x) <= cfg.end_date)
        prev_day, prev_eq, month_start_eq = None, cfg.start_equity, cfg.start_equity
        for i in range(i0, i1 + 1):
            d = days[i]
            if prev_day is not None:
                self.cash += self.cash * cfg.apy / 365.0 * (d - prev_day).days
                if d.month != prev_day.month:
                    if cfg.cb_days > 0 and rules.cb_trigger(month_start_eq, prev_eq):
                        self.cb_until = i + cfg.cb_days
                    month_start_eq = prev_eq
            try:
                bars = self.p.get_underlying_bars(d, cfg.resolution)
                ts, spots = bars.index, bars["spot"].tolist()
            except MissingData:      # no intraday underlying: nothing can be checked today
                ts, spots = pd.DatetimeIndex([]), []
                self.gaps += 1
            mins = [t.hour * 60 + t.minute for t in ts]
            for pos in self.pos:
                if len(ts):
                    self._load_day(pos, d, ts)
                else:
                    pos.q = None
            entry_bars: list[int] = []
            if (i - i0) % cfg.entry_every == 0:
                for m in entry_minutes(d, cfg):
                    bb = _bar_for(mins, m)
                    if bb is not None and bb not in entry_bars:
                        entry_bars.append(bb)
            first_reason, entered = None, False
            for b in range(len(ts)):
                for pos in list(self.pos):
                    if pos.q is None or (pos.entry_day == d and b <= pos.entry_bar):
                        continue
                    quotes = [(pos.q[k][0][b], pos.q[k][1][b]) for k in range(len(pos.legs))]
                    if not all(rules.valid_quote(bq, aq) for bq, aq in quotes):
                        continue
                    nat = rules.close_cost(pos.legs, quotes, cfg, "natural")
                    fc = rules.close_cost(pos.legs, quotes, cfg)
                    pos.last_mid = rules.close_cost(pos.legs, quotes, cfg, "mid")
                    if pos.kind == "condor":
                        why = rules.check_exit(pos.credit, nat, fc, spots[b], pos.sp, pos.sc,
                                               (pos.expiry - d).days, pos.entry_dte, cfg)
                        if why:
                            self._close(pos, why, fc, ts[b], d)
                if b in entry_bars and not entered:
                    why = self._try_entry(i, d, b, ts, mins, spots)
                    if why is None:
                        entered = True
                    elif first_reason is None:
                        first_reason = why
            if entry_bars and not entered and first_reason:
                self._block(first_reason)
            for pos in list(self.pos):       # expiry settlement on the SPY close
                if pos.expiry <= d:
                    cost = rules.settle_cost(pos.sp, pos.lp, pos.sc, pos.lc, self.close[i])
                    self._close(pos, "expiry", cost, pd.Timestamp(d) + pd.Timedelta(hours=16), d)
            prev_day, prev_eq = d, self.equity()
            self.eq_rows.append({"date": d, "cash": round(self.cash, 2), "equity": round(prev_eq, 2),
                                 "open_risk": round(sum(p.risk for p in self.pos), 2),
                                 "n_open": len(self.pos)})
        for pos in list(self.pos):           # leave nothing open: mark out at last mid
            self._close(pos, "end", pos.last_mid, pd.Timestamp(prev_day), prev_day)
        if self.eq_rows:
            self.eq_rows[-1]["equity"] = round(self.cash, 2)
        return Result(cfg, self.p.name, pd.DataFrame(self.trades), pd.DataFrame(self.eq_rows),
                      self.blocked, self.gaps, self.ev_note)


def run_backtest(provider: Provider, cfg: Config, events: pd.DataFrame | None = None,
                 events_note: str = "") -> Result:
    if cfg.event_buffer and events is None:
        raise ValueError("event_buffer is on but no events were supplied")
    return _Run(provider, cfg, events if cfg.event_buffer or cfg.earnings_filter else None,
                events_note).run()
