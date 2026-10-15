"""Combination x allocation grid, ranking, verdicts, parquet output."""
from __future__ import annotations

import itertools
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import ALLOCATIONS, LEVEL_TAGS, START_EQUITY
from . import montecarlo as mcm
from . import thermo, verdicts
from .replay import ReplayConfig, build_variant, max_drawdown, monthly_stats, replay

MIX = "all_strategies_equal"
STRESS_RE = re.compile(r"^stress_", re.I)


@dataclass
class GridConfig:
    allocations: list = field(default_factory=lambda: list(ALLOCATIONS))
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    mc: bool = True
    paths: int = 10000
    block: int = 2
    seed: int = 7
    horizons: tuple = (12, 36)
    mixes: bool = True
    jobs: int = 1
    thermo_cfg: dict | None = None
    max_tickers_subsets: int = 5


def _slug(*parts) -> str:
    return "|".join(str(p) for p in parts)


def scenario_id(profile, config_tag, strategy, tickers, alloc) -> str:
    return _slug(profile, config_tag, strategy, "+".join(tickers), f"A{int(alloc)}")


def subsets(items):
    items = sorted(items)
    for r in range(1, len(items) + 1):
        for c in itertools.combinations(items, r):
            yield list(c)


def groups(trades: pd.DataFrame, include_stress=False):
    """(profile, config_tag) variants - stress windows are excluded from the 10-year grid."""
    for (p, c), g in trades.groupby(["profile", "config_tag"], sort=True):
        if not include_stress and STRESS_RE.match(str(p)):
            continue
        yield p, c, g


def _one_group(args):
    (profile, tag, gtr, gmarks, gcfg) = args
    rows, series = [], {}
    days = pd.DatetimeIndex(sorted(pd.concat([gtr["entry_ts"].dt.normalize(), gtr["exit_ts"].dt.normalize(),
                                              gmarks["date"] if len(gmarks) else pd.Series([], dtype="datetime64[ns]")]).dropna().unique()))
    level = int(gtr["level"].mode().iloc[0])
    strategies = sorted(gtr["strategy"].unique())
    sets = [(s, [s]) for s in strategies]
    if gcfg.mixes and len(strategies) > 1:
        sets.append((MIX, strategies))
    tcfg = gcfg.thermo_cfg or thermo.load_config()
    for sname, slist in sets:
        sub = gtr[gtr["strategy"].isin(slist)]
        vd = build_variant(sub, gmarks, days)
        tick_all = sorted(sub["ticker"].unique())
        tk = vd.trades["ticker"].to_numpy()
        for T in subsets(tick_all):
            mask = np.isin(tk, T)
            if not mask.any():
                continue
            vixmode = vd.trades.loc[mask, "vix_bucket"].mode()
            delta = vd.trades.loc[mask, "short_put_delta"].median()
            for A in gcfg.allocations:
                res = replay(vd, mask, A / 100.0, gcfg.replay)
                m = dict(res.metrics)
                dd = max_drawdown(res.equity, gcfg.replay.start_equity)
                ret, low, _, _ = monthly_stats(res.equity, vd.month_id, gcfg.replay.start_equity)
                m["dd_trough_date"] = str(vd.days[int(dd.argmax())].date())
                if gcfg.mc:
                    m.update(mcm.run_mc(ret, low, gcfg.replay.start_equity, m["min_collateral"], gcfg.horizons,
                                        gcfg.block, gcfg.paths, gcfg.seed))
                    ruin = m.get(f"mc{tcfg['ruin_horizon_months']}_p_broke", m.get(f"mc{gcfg.horizons[-1]}_p_broke"))
                else:
                    ruin = None
                m["ruin_odds"] = ruin
                m["zone"] = thermo.zone(m["max_dd"], ruin, m["broke"], tcfg)
                sid = scenario_id(profile, tag, sname, T, A)
                m.update(scenario_id=sid, level=level, level_tag=LEVEL_TAGS.get(level, str(level)), profile=profile,
                         config_tag=tag, strategy=sname, tickers="+".join(T), n_tickers=len(T), allocation=A,
                         typical_vix=(vixmode.iloc[0] if len(vixmode) else ""), short_delta=float(delta) if pd.notna(delta) else np.nan,
                         group=f"{profile}|{tag}|{sname}")
                rows.append(m)
                series[sid] = {"eq": res.equity.astype(np.float32), "days": vd.days, "gkey": f"{profile}|{tag}|{sname}"}
    return rows, series


def run_grid(trades: pd.DataFrame, marks: pd.DataFrame, gcfg: GridConfig | None = None, progress=None):
    gcfg = gcfg or GridConfig()
    tasks = []
    for p, c, g in groups(trades):
        mk = marks[marks["uid"].isin(set(g["uid"]))] if len(marks) else marks
        tasks.append((p, c, g, mk, gcfg))
    rows, series = [], {}
    if gcfg.jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=gcfg.jobs) as ex:
            outs = list(ex.map(_one_group, tasks))
    else:
        outs = []
        for t in tasks:
            outs.append(_one_group(t))
            if progress:
                progress(f"  grid {t[0]}/{t[1]}")
    for r, s in outs:
        rows += r
        series.update(s)
    df = pd.DataFrame(rows)
    return df, series


# ------------------------------------------------------------------ ranking + verdict text
def rank(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main list: NOT-broke scenarios by return/maxDD (evidence-grade rows first); broke rows listed separately,
    ranked by how late they went broke."""
    ok = df[~df["broke"]].copy()
    ok = ok.sort_values(["too_few_trades", "calmar", "max_dd"], ascending=[True, False, True], kind="stable")
    ok.insert(0, "rank", np.arange(1, len(ok) + 1))
    br = df[df["broke"]].copy()
    br = br.sort_values(["broke_date", "final_equity"], ascending=[False, False], kind="stable")
    br.insert(0, "rank", np.arange(1, len(br) + 1))
    return ok, br


def add_verdicts(df: pd.DataFrame, vcfg: dict) -> pd.DataFrame:
    df = df.copy()
    fam = {}
    for g, sub in df[df["broke"]].groupby(["profile", "config_tag", "strategy", "tickers"]):
        fam[g] = {int(a): d for a, d in zip(sub["allocation"], sub["broke_date"])}
    out = []
    for r in df.to_dict("records"):
        f = fam.get((r["profile"], r["config_tag"], r["strategy"], r["tickers"]))
        out.append(verdicts.verdict(r, vcfg, f))
    df["verdict"] = out
    return df


def select_for_parquet(ok: pd.DataFrame, br: pd.DataFrame, top=50, bottom=20) -> list:
    ids = list(ok["scenario_id"].head(top)) + list(ok["scenario_id"].tail(bottom)) + list(br["scenario_id"])
    return list(dict.fromkeys(ids))


def safe_name(sid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", sid)


def write_parquet(out_dir, ids: list, series: dict, start_eq=START_EQUITY) -> int:
    from pathlib import Path
    d = Path(out_dir) / "equity"
    d.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        s = series[sid]
        eq = s["eq"].astype(float)
        pd.DataFrame({"date": s["days"], "equity": eq, "drawdown": max_drawdown(eq, start_eq)}).to_parquet(
            d / f"{safe_name(sid)}.parquet", index=False, compression="zstd")
    return len(ids)


# ------------------------------------------------------------------ VIX-regime lookup
VIX_BUCKETS = ["lt15", "15to20", "20to30", "gt30"]


def _vix_group(args):
    profile, tag, gtr, gmarks, gcfg = args
    out = []
    days = pd.DatetimeIndex(sorted(pd.concat([gtr["entry_ts"].dt.normalize(), gtr["exit_ts"].dt.normalize(),
                                              gmarks["date"] if len(gmarks) else pd.Series([], dtype="datetime64[ns]")]).dropna().unique()))
    tcfg = gcfg.thermo_cfg or thermo.load_config()
    strategies = sorted(gtr["strategy"].unique())
    for st in strategies:
        sub = gtr[gtr["strategy"] == st]
        vd = build_variant(sub, gmarks, days)
        for b in VIX_BUCKETS:
            mask = (vd.trades["vix_bucket"] == b).to_numpy()
            if mask.sum() < 1:
                continue
            delta = vd.trades.loc[mask, "short_put_delta"].median()
            for A in gcfg.allocations:
                res = replay(vd, mask, A / 100.0, gcfg.replay)
                m = res.metrics
                ruin = None
                if gcfg.mc:
                    ret, low, _, _ = monthly_stats(res.equity, vd.month_id, gcfg.replay.start_equity)
                    mc = mcm.run_mc(ret, low, gcfg.replay.start_equity, m["min_collateral"], (tcfg["ruin_horizon_months"],),
                                    gcfg.block, gcfg.paths, gcfg.seed)
                    ruin = mc[f"mc{tcfg['ruin_horizon_months']}_p_broke"]
                z = thermo.zone(m["max_dd"], ruin, m["broke"], tcfg)
                out.append({"bucket": b, "profile": profile, "config_tag": tag, "strategy": st, "allocation": A, "zone": z,
                            "short_delta": float(delta) if pd.notna(delta) else np.nan})
    return out


def vix_regime_lookup(trades: pd.DataFrame, marks: pd.DataFrame, gcfg: GridConfig | None = None) -> pd.DataFrame:
    """Per VIX bucket at entry: trade stats, plus the highest allocation / widest delta whose regime-only replay
    (entries only when VIX was in that bucket; all tickers of the log) stayed in Safe or Mild."""
    gcfg = gcfg or GridConfig()
    t = trades[~trades["profile"].str.match(STRESS_RE.pattern, case=False)]
    tasks = []
    for p, c, g in groups(t):
        mk = marks[marks["uid"].isin(set(g["uid"]))] if len(marks) else marks
        tasks.append((p, c, g, mk, gcfg))
    recs = []
    if gcfg.jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=gcfg.jobs) as ex:
            for o in ex.map(_vix_group, tasks):
                recs += o
    else:
        for tk in tasks:
            recs += _vix_group(tk)
    rr = pd.DataFrame(recs)
    rows = []
    for b in VIX_BUCKETS:
        g = t[t["vix_bucket"] == b]
        pc = g["pnl_per_contract"] / g["collateral_per_contract"]
        row = {"bucket": b, "trades": len(g), "win_rate": float((g["pnl_per_contract"] > 0).mean()) if len(g) else np.nan,
               "avg_pnl_pct_collateral": float(pc.mean()) if len(g) else np.nan,
               "worst_pnl_pct_collateral": float(pc.min()) if len(g) else np.nan,
               "thin_evidence": len(g) < 30, "max_alloc_safe_or_mild": None, "widest_delta_safe_or_mild": None, "where": ""}
        if len(rr):
            good = rr[(rr["bucket"] == b) & rr["zone"].isin(["Safe", "Mild"])]
            if len(good):
                row["max_alloc_safe_or_mild"] = int(good["allocation"].max())
                gd = good[good["strategy"].isin(["iron_condor", "bull_put_spread"])].dropna(subset=["short_delta"])   # delta is only comparable for fixed-delta shorts
                if len(gd):
                    wi = gd.loc[gd["short_delta"].idxmax()]
                    row["widest_delta_safe_or_mild"] = float(wi["short_delta"])
                    row["where"] = f"{wi['profile']}/{wi['config_tag']} {wi['strategy']}"
        rows.append(row)
    return pd.DataFrame(rows)
