"""Stress lab: stress-window table, slippage through stops, breaking-point finder."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import montecarlo as mcm
from .replay import ReplayConfig, build_variant, max_drawdown, monthly_stats, replay

STOP_REASONS = ("stop", "stop_dollar")
PROTECTIVE = ("stop", "stop_dollar", "breach", "trigger_exit")


def stop_levels(t: pd.DataFrame, cfgs: dict) -> np.ndarray:
    """Stop trigger (USD per contract, negative). Uses column stop_level_per_contract if present, else the run's
    config.json (stop_dollar | stop_multiple [+ stop_floor_width_frac]); NaN when unknowable."""
    if "stop_level_per_contract" in t.columns:
        lv = pd.to_numeric(t["stop_level_per_contract"], errors="coerce").to_numpy(float)
        return -np.abs(lv)
    out = np.full(len(t), np.nan)
    for rid, idx in t.groupby("run_id").indices.items():
        c = cfgs.get(rid, {}) or {}
        sub = t.iloc[idx]
        if c.get("stop_dollar") is not None:
            out[idx] = -abs(float(c["stop_dollar"]))
        elif c.get("stop_multiple") is not None:
            fl = float(c.get("stop_floor_width_frac") or 0.0)
            out[idx] = -np.maximum(float(c["stop_multiple"]) * sub["credit_per_contract"].to_numpy(float),
                                   fl * sub["width"].to_numpy(float) * 100.0)
    return out


def stress_table(trades: pd.DataFrame, marks: pd.DataFrame, cfgs: dict) -> pd.DataFrame:
    sel = trades[trades["profile"].str.match(r"^(stress_|experiment_)", case=False)].copy()
    if sel.empty:
        return pd.DataFrame()
    sel["stop_level"] = stop_levels(sel, cfgs)
    wm = marks[marks["uid"].isin(set(sel["uid"]))].groupby("uid")["mark_pnl_per_contract"].min() if len(marks) else pd.Series(dtype=float)
    sel["worst_mark"] = sel["uid"].map(wm)
    gross = sel["pnl_per_contract"] + sel["fees_per_contract"]
    is_stop = sel["exit_reason"].isin(STOP_REASONS)
    sel["thru_stop"] = is_stop & (gross < sel["stop_level"] * 1.02 - 0.01)   # >2% beyond the trigger
    sel["thru_amt"] = np.where(sel["thru_stop"], sel["stop_level"] - gross, 0.0)
    rows = []
    for (p, tag, st), g in sel.groupby(["profile", "config_tag", "strategy"]):
        ns = int(is_stop[g.index].sum())
        rows.append({
            "window": p, "config_tag": tag, "strategy": st, "kind": "stress window" if p.startswith("stress_") else "experiment",
            "trades": len(g), "pnl_mean": g["pnl_per_contract"].mean(), "pnl_worst": g["pnl_per_contract"].min(),
            "pnl_total": g["pnl_per_contract"].sum(), "worst_mark": g["worst_mark"].min(),
            "exit_reasons": ", ".join(f"{k}:{v}" for k, v in g["exit_reason"].value_counts().items()),
            "stops": ns, "stops_through_trigger": int(g["thru_stop"].sum()),
            "slippage_through_stop": ("n/a (no stop level)" if g["stop_level"].isna().all() else
                                      (f"{int(g['thru_stop'].sum())} of {ns} stops filled through the trigger"
                                       + (f" (avg ${g.loc[g['thru_stop'], 'thru_amt'].mean():.0f}/contract beyond)" if g["thru_stop"].any() else ""))),
        })
    return pd.DataFrame(rows).sort_values(["kind", "window", "strategy", "config_tag"]).reset_index(drop=True)


# --------------------------------------------------------------------------- breaking point
def _fail(res, dd_limit):
    dd = max_drawdown(res.equity, res.start_equity)
    over = np.where(dd > dd_limit)[0]
    first_dd = int(over[0]) if len(over) else None
    cand = [x for x in (first_dd, res.broke_idx) if x is not None]
    return (min(cand) if cand else None), dd


def breaking_point(vd, mask, cfg: ReplayConfig | None = None, dd_limit=0.5, a_max=100, bisect=False, a_min=1):
    """Raise allocation (1% steps) until max drawdown > dd_limit or the account goes broke."""
    cfg = cfg or ReplayConfig()
    scan = []

    def run(a):
        res = replay(vd, mask, a / 100.0, cfg)
        fi, dd = _fail(res, dd_limit)
        scan.append({"allocation": a, "max_dd": float(dd.max()), "broke": res.broke_idx is not None})
        return res, fi

    first = None
    if bisect:
        lo, hi = a_min, a_max
        rhi, fhi = run(hi)
        if fhi is not None:
            rlo, flo = run(lo)
            if flo is not None:
                first = (lo, rlo, flo)
            else:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    r, f = run(mid)
                    if f is None:
                        lo = mid
                    else:
                        hi, rhi, fhi = mid, r, f
                first = (hi, rhi, fhi)
    else:
        for a in range(a_min, a_max + 1):
            r, f = run(a)
            if f is not None:
                first = (a, r, f)
                break
    if first is None:
        return {"found": False, "scan": scan, "limit": dd_limit, "a_max": a_max}
    a, res, fi = first
    d0 = vd.days[fi]
    kind = "broke" if res.broke_idx is not None and res.broke_idx == fi else f"drawdown > {int(dd_limit * 100)}%"
    t = vd.trades
    acc = res.acc
    usd = res.contracts[acc] * vd.pnl[acc]
    xi = vd.xi[acc]
    sel = acc[(xi <= fi) & (xi >= fi - 45)] if len(acc) else acc
    sel_usd = res.contracts[sel] * vd.pnl[sel]
    order = np.argsort(sel_usd)[:5]
    causes = [{"trade_id": t.loc[sel[o], "trade_id"], "ticker": t.loc[sel[o], "ticker"], "strategy": t.loc[sel[o], "strategy"],
               "exit_ts": str(t.loc[sel[o], "exit_ts"]), "exit_reason": t.loc[sel[o], "exit_reason"],
               "contracts": int(res.contracts[sel[o]]), "usd": float(sel_usd[o])} for o in order if sel_usd[o] < 0]
    return {"found": True, "allocation": a, "kind": kind, "date": str(d0.date()), "causes": causes, "scan": scan,
            "limit": dd_limit, "max_dd_at_alloc": float(max_drawdown(res.equity, res.start_equity).max())}


def first_delta_over(trades: pd.DataFrame, marks: pd.DataFrame, strategy: str, tickers: list, alloc=15,
                     threshold=0.05, cfg: ReplayConfig | None = None, paths=10000, block=2, seed=7, horizon=36):
    """Across delta settings found in the logs (non-stress, same strategy+tickers), ascending by short delta:
    the first whose Monte Carlo P(broke) exceeds the threshold."""
    cfg = cfg or ReplayConfig()
    sub = trades[(trades["strategy"] == strategy) & trades["ticker"].isin(tickers) & ~trades["profile"].str.match("^stress_", case=False)]
    rows = []
    for (p, tag), g in sub.groupby(["profile", "config_tag"]):
        if set(tickers) - set(g["ticker"]):
            continue
        d = g["short_put_delta"].median()
        if pd.isna(d):
            continue
        vd = build_variant(g, marks)
        mask = np.ones(vd.n, bool)
        res = replay(vd, mask, alloc / 100.0, cfg)
        ret, low, _, _ = monthly_stats(res.equity, vd.month_id, cfg.start_equity)
        mc = mcm.run_mc(ret, low, cfg.start_equity, res.metrics["min_collateral"], (horizon,), block, paths, seed)
        rows.append({"profile": p, "config_tag": tag, "short_delta": float(d), "p_broke": mc[f"mc{horizon}_p_broke"],
                     "max_dd": res.metrics["max_dd"], "replay_broke": res.metrics["broke"]})
    tab = pd.DataFrame(rows).sort_values("short_delta").reset_index(drop=True) if rows else pd.DataFrame()
    hit = tab[tab["p_broke"] > threshold] if len(tab) else tab
    return tab, (hit.iloc[0].to_dict() if len(hit) else None)
