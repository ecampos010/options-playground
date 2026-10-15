"""Portfolio replay simulator: per-contract unit logs -> account equity.

Sizing is path dependent (budget = allocation x CURRENT equity), so the loop is sequential over days; marks come
from a compact per-trade matrix so drawdowns include open-position losses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import floor

import numpy as np
import pandas as pd

from . import START_EQUITY

ENTRY, EXIT = 1, 0


@dataclass
class ReplayConfig:
    start_equity: float = START_EQUITY
    apy: float = 0.036
    expiry_cap: float | None = 0.60     # max share of a sleeve's budget reserved on ONE expiry; None = off
    cb_days: int = 0                    # 0 = off, 21 = pause entries ~21 trading days after a losing calendar month
    min_trades_evidence: int = 30


@dataclass
class VariantData:
    """Everything the replay needs for a set of trades sharing one calendar (so scenarios are paired)."""
    trades: pd.DataFrame            # sorted by entry_ts, reset index; columns incl. uid ticker strategy
    days: pd.DatetimeIndex
    ei: np.ndarray                  # entry day index
    xi: np.ndarray                  # exit day index
    coll: np.ndarray
    pnl: np.ndarray
    expiry_ord: np.ndarray
    Mc: np.ndarray                  # (n, K) mark by days-since-entry, forward filled, float32
    events: list                    # per day: list of (kind, i) in time order
    gaps: np.ndarray                # calendar days since previous calendar day
    month_id: np.ndarray
    sleeve_key: np.ndarray          # object array of (ticker, strategy)

    @property
    def n(self):
        return len(self.trades)


def build_variant(trades: pd.DataFrame, marks: pd.DataFrame, days: pd.DatetimeIndex | None = None) -> VariantData:
    t = trades.sort_values(["entry_ts", "uid"], kind="stable").reset_index(drop=True)
    n = len(t)
    ed = t["entry_ts"].dt.normalize()
    xd = t["exit_ts"].dt.normalize()
    mk = marks[marks["uid"].isin(set(t["uid"]))] if len(marks) else marks
    if days is None:
        parts = [ed, xd]
        if len(mk):
            parts.append(mk["date"])
        days = pd.DatetimeIndex(sorted(pd.concat(parts).dropna().unique()))
    ei = days.get_indexer(ed)
    xi = days.get_indexer(xd)
    uid_ix = pd.Series(np.arange(n), index=t["uid"].values)
    if len(mk):
        mi = uid_ix.reindex(mk["uid"].values).to_numpy()
        md = days.get_indexer(mk["date"])
        ok = (md >= 0) & ~np.isnan(mi)
        mi, md, mv = mi[ok].astype(int), md[ok], mk["mark_pnl_per_contract"].to_numpy()[ok]
        k = md - ei[mi]
        ok2 = k >= 0
        mi, k, mv = mi[ok2], k[ok2], mv[ok2]
        K = int(max(k.max() + 1 if len(k) else 1, (xi - ei).max() + 1))
    else:
        mi = k = np.array([], int); mv = np.array([])
        K = int((xi - ei).max() + 1)
    Mc = np.full((n, K), np.nan, dtype=np.float32)
    Mc[mi, k] = mv
    # forward fill along the life axis; leading gaps -> 0
    have = ~np.isnan(Mc)
    idx = np.where(have, np.arange(K)[None, :], -1)
    np.maximum.accumulate(idx, axis=1, out=idx)
    filled = np.take_along_axis(Mc, np.maximum(idx, 0), axis=1)
    Mc = np.where(idx >= 0, filled, 0.0).astype(np.float32)
    # events
    D = len(days)
    ev: list = [[] for _ in range(D)]
    ets = t["entry_ts"].to_numpy()
    xts = t["exit_ts"].to_numpy()
    for i in range(n):
        ev[ei[i]].append((ets[i], ENTRY, ENTRY, i))
        ev[xi[i]].append((xts[i], EXIT, EXIT, i))
    ev = [[(kind, i) for (_, _, kind, i) in sorted(e, key=lambda r: (r[0], r[1]))] for e in ev]
    gaps = np.ones(D)
    gaps[1:] = np.diff(days.values).astype("timedelta64[D]").astype(int)
    gaps[0] = 0
    month_id = (days.year * 12 + days.month).to_numpy()
    exp_ord = pd.factorize(t["expiry"])[0]
    sk = np.empty(n, dtype=object)
    sk[:] = list(zip(t["ticker"], t["strategy"]))
    return VariantData(t, days, ei, xi, t["collateral_per_contract"].to_numpy(float), t["pnl_per_contract"].to_numpy(float),
                       exp_ord, Mc, ev, gaps, month_id, sk)


@dataclass
class Result:
    equity: np.ndarray
    metrics: dict
    acc: np.ndarray            # accepted trade indices in exit order
    contracts: np.ndarray      # per trade index (0 if not taken)
    eq_entry: np.ndarray       # equity at entry (per trade index)
    broke_idx: int | None
    days: pd.DatetimeIndex
    month_id: np.ndarray
    start_equity: float
    skipped: dict = field(default_factory=dict)


def sleeve_weights(vd: VariantData, mask: np.ndarray) -> np.ndarray:
    """Equal split across tickers, then equally across the strategies that exist for that ticker."""
    w = np.zeros(vd.n)
    tick = vd.trades["ticker"].to_numpy()
    strat = vd.trades["strategy"].to_numpy()
    sel = np.where(mask)[0]
    if len(sel) == 0:
        return w
    df = pd.DataFrame({"t": tick[sel], "s": strat[sel]}).drop_duplicates()
    nt = df["t"].nunique()
    per_t = df.groupby("t")["s"].nunique()
    wmap = {(r.t, r.s): 1.0 / nt / per_t[r.t] for r in df.itertuples()}
    for i in sel:
        w[i] = wmap[(tick[i], strat[i])]
    return w


def replay(vd: VariantData, mask: np.ndarray, alloc: float, cfg: ReplayConfig | None = None,
           weights: np.ndarray | None = None) -> Result:
    """alloc: fraction of CURRENT equity at risk in total (0.15 = 15%). mask: which trades are in play."""
    cfg = cfg or ReplayConfig()
    n, D = vd.n, len(vd.days)
    w = sleeve_weights(vd, mask) if weights is None else weights
    share = (alloc * w).tolist()
    active = mask.tolist()
    coll = vd.coll.tolist()
    pnlu = vd.pnl.tolist()
    ei = vd.ei
    sk = vd.sleeve_key.tolist()
    exp_ord = vd.expiry_ord.tolist()
    Mc = vd.Mc
    min_coll = float(vd.coll[mask].min()) if mask.any() else 0.0
    cash = cfg.start_equity
    reserved = 0.0
    resv_s: dict = {}
    resv_e: dict = {}
    contracts = np.zeros(n)
    c_list = [0] * n
    eq_entry = np.zeros(n)
    open_list: list = []
    acc_exit: list = []
    eq = np.zeros(D)
    broke_idx = None
    pause_until = -1
    cb_ref = cfg.start_equity
    sk_cant = sk_cb = sk_cap = 0
    rate = cfg.apy
    ev = vd.events
    gaps = vd.gaps
    mid = vd.month_id
    for d in range(D):
        if d and gaps[d] > 0 and cash > reserved and rate:
            cash += (cash - reserved) * ((1.0 + rate) ** (gaps[d] / 365.0) - 1.0)
        if cfg.cb_days and d and mid[d] != mid[d - 1]:
            prev_end = eq[d - 1]
            if prev_end < cb_ref:
                pause_until = d + cfg.cb_days
            cb_ref = prev_end
        for kind, i in ev[d]:
            if not active[i]:
                continue
            if kind == EXIT:
                c = c_list[i]
                if c:
                    cash += c * pnlu[i]
                    if cash < 0:
                        cash = 0.0
                    r = c * coll[i]
                    reserved -= r
                    resv_s[sk[i]] -= r
                    resv_e[(sk[i], exp_ord[i])] -= r
                    open_list.remove(i)
                    acc_exit.append(i)
                continue
            # ENTRY
            if broke_idx is not None:
                continue
            if d < pause_until:
                sk_cb += 1
                continue
            if open_list:
                idx = np.array(open_list)
                kk = d - 1 - ei[idx]
                okm = kk >= 0
                mt = float(Mc[idx[okm], kk[okm]] @ contracts[idx[okm]]) if okm.any() else 0.0
            else:
                mt = 0.0
            e = cash + mt
            if e < 0:
                e = 0.0
            amt = share[i] * e
            cl = coll[i]
            n_t = floor(amt / cl + 1e-9)
            nn = n_t
            if n_t >= 1:
                if cfg.expiry_cap is not None:
                    capamt = cfg.expiry_cap * amt - resv_e.get((sk[i], exp_ord[i]), 0.0)
                    nn = min(nn, floor(capamt / cl + 1e-9))
                nn = min(nn, floor((amt - resv_s.get(sk[i], 0.0)) / cl + 1e-9), floor((e - reserved) / cl + 1e-9))
            if nn < 1:
                sk_cant += 1
                continue
            contracts[i] = nn
            c_list[i] = nn
            eq_entry[i] = e
            r = nn * cl
            reserved += r
            resv_s[sk[i]] = resv_s.get(sk[i], 0.0) + r
            resv_e[(sk[i], exp_ord[i])] = resv_e.get((sk[i], exp_ord[i]), 0.0) + r
            open_list.append(i)
        if open_list:
            idx = np.array(open_list)
            kk = d - ei[idx]
            mt = float(Mc[idx, np.minimum(kk, Mc.shape[1] - 1)] @ contracts[idx])
        else:
            mt = 0.0
        e = cash + mt
        if e < 0:
            e = 0.0
        eq[d] = e
        if broke_idx is None and (e <= 0 or e < min_coll):
            broke_idx = d
    acc = np.array(acc_exit, int)
    m = compute_metrics(vd, eq, acc, contracts, eq_entry, broke_idx, cfg, min_coll, mask)
    m.update(skipped_cant_afford=sk_cant, skipped_cb=sk_cb)
    return Result(eq, m, acc, contracts, eq_entry, broke_idx, vd.days, vd.month_id, cfg.start_equity,
                  {"cant_afford": sk_cant, "cb": sk_cb})


def monthly_stats(eq: np.ndarray, month_id: np.ndarray, start_eq: float):
    """Per calendar month: return, and the worst equity inside the month relative to month-start equity."""
    D = len(eq)
    first = np.r_[True, month_id[1:] != month_id[:-1]]
    starts = np.where(first)[0]
    ends = np.r_[starts[1:], D] - 1
    prev = np.r_[start_eq, eq[ends[:-1]]]
    endv = eq[ends]
    lows = np.minimum.reduceat(eq, starts)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(prev > 0, endv / prev - 1.0, 0.0)
        low = np.where(prev > 0, lows / prev - 1.0, 0.0)
    return ret, np.minimum(low, ret), month_id[starts], endv


def max_drawdown(eq: np.ndarray, start_eq: float):
    peak = np.maximum.accumulate(np.maximum(eq, start_eq))
    dd = np.where(peak > 0, 1.0 - eq / peak, 0.0)
    return dd


def compute_metrics(vd, eq, acc, contracts, eq_entry, broke_idx, cfg, min_coll, mask) -> dict:
    s0 = cfg.start_equity
    days = vd.days
    years = max((days[-1] - days[0]).days / 365.25, 1e-9) if len(days) > 1 else 1e-9
    final = float(eq[-1])
    cagr = (final / s0) ** (1.0 / years) - 1.0 if final > 0 else -1.0
    dd = max_drawdown(eq, s0)
    ret, low, _, _ = monthly_stats(eq, vd.month_id, s0)
    usd = contracts[acc] * vd.pnl[acc] if len(acc) else np.array([])
    pct = usd / eq_entry[acc] if len(acc) else np.array([])
    wins = usd[usd > 0].sum() if len(usd) else 0.0
    losses = -usd[usd < 0].sum() if len(usd) else 0.0
    streak = best = 0
    for v in (vd.pnl[acc] if len(acc) else []):
        streak = streak + 1 if v < 0 else 0
        best = max(best, streak)
    mdd = float(dd.max()) if len(dd) else 0.0
    ntr = int(len(acc))
    return {
        "final_equity": final, "cagr": cagr, "max_dd": mdd, "worst_month": float(ret.min()) if len(ret) else 0.0,
        "worst_trade_usd": float(usd.min()) if ntr else 0.0, "worst_trade_pct": float(pct.min()) if ntr else 0.0,
        "win_rate": float((vd.pnl[acc] > 0).mean()) if ntr else 0.0,
        "profit_factor": float(wins / losses) if losses > 0 else (np.inf if wins > 0 else 0.0),
        "calmar": float(cagr / max(mdd, 0.005)),
        "avg_contracts": float(contracts[acc].mean()) if ntr else 0.0,
        "longest_losing_streak": int(best), "broke": broke_idx is not None,
        "broke_date": str(days[broke_idx].date()) if broke_idx is not None else "",
        "years": float(years), "n_trades": ntr, "too_few_trades": ntr < cfg.min_trades_evidence,
        "min_collateral": min_coll,
    }
