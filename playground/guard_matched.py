"""Matched guarded-vs-plain comparison on the lab engine: same entry days, same exits, guards on vs off.

PLAIN    no entry guards. Iron condor, short strikes at delta 0.16 (both sides), DTE target 45, take-profit 50% of
         credit, stop at 2x credit on the natural buyback, breach exit, 21-DTE time exit, natural fills, Green's
         15% budget (both arms share it; the `experiment` tier would silently use 5%).
GUARDED  PLAIN + the Green entry guards that exist in code (rules.py / config.py defaults):
         gate (VIX >= 25 or SPY day <= -1.5% or SPY 5-day <= -4%), upside (5-day >= +4%),
         event buffer (FOMC, CPI, PPI, NFP, ELECTION: no entry on the event day or 1-2 trading days before),
         circuit breaker (21 trading days after a losing month), earnings filter (inert for SPY: no EARNINGS rows).

The guarded arm may enter only the days the plain arm entered (run_lab(entry_days=...)), so its entries are a
strict subset of the plain arm's by construction. Sizing and the equity path are each arm's own.

Usage (VPS, reads the existing parquet cache only; nothing is downloaded):
    python -m playground.guard_matched --provider theta --data-dir data --events data/events.csv
    python -m playground.guard_matched --provider synthetic --start 2016-10-01 --end 2017-03-31   (plumbing only)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import factory
from .config import Config, diff_switches
from .events import load_events
from .lab import LabResult, entry_schedule, run_lab

GUARD_KEYS = ("gate", "upside_guard", "event_buffer", "cb_days", "earnings_filter")
GUARD_REASONS = ("gate", "upside", "event", "cb")
GAP_REASONS = ("nodata", "noquote")
EXITS = ("stop", "tp", "time", "breach", "expiry", "other")
PLAIN_PROFILE, GUARDED_PROFILE = "matched_plain", "matched_guarded"


# ---------------------------------------------------------------- the two arms
def _arm(profile: str, cadence: str, start: str, end: str, guards: bool) -> Config:
    green = Config()
    return Config(
        profile=profile, tier="green", ticker="SPY", strategy="iron_condor", sizing="equity-pct",
        entry_cadence=cadence, dte_target=45,
        strike_rule={**green.strike_rule, "mode": "delta", "put_delta": 0.16, "call_delta": 0.16},
        tp=0.50, stop_mult=2.0, breach_exit=True, time_exit=21, fill="natural",
        resolution="5m", spot_source="alpaca", spot_fallback="parity", start_date=start, end_date=end,
        gate=guards, upside_guard=guards, event_buffer=guards, cb_days=green.cb_days if guards else 0,
        earnings_filter=guards)


def plain_config(cadence: str, start: str, end: str) -> Config:
    return _arm(PLAIN_PROFILE, cadence, start, end, guards=False)


def guarded_config(cadence: str, start: str, end: str) -> Config:
    return _arm(GUARDED_PROFILE, cadence, start, end, guards=True)


@dataclass
class Matched:
    cadence: str
    plain: LabResult
    guarded: LabResult
    scheduled_days: int


def entry_dates(res: LabResult) -> set[date]:
    """Dates on which the arm opened a position, including positions still open when the data ends."""
    stamps = list(res.trades.entry_ts) + [x["entry_ts"] for x in res.meta["open_excluded"]["trades"]]
    return {pd.Timestamp(s).date() for s in stamps}


def run_matched(provider, events, events_note: str, cadence: str, start: str, end: str) -> Matched:
    plain = run_lab(provider, plain_config(cadence, start, end), events, events_note)
    guarded = run_lab(provider, guarded_config(cadence, start, end), events, events_note,
                      entry_days=entry_dates(plain))
    return Matched(cadence, plain, guarded, _scheduled_days(provider, plain_config(cadence, start, end)))


def _scheduled_days(provider, cfg: Config) -> int:
    days = list(provider.get_underlying_daily().index)
    i0 = max(6, next((j for j, x in enumerate(days) if str(x) >= cfg.start_date), len(days)))
    i1 = max(j for j, x in enumerate(days) if str(x) <= cfg.end_date)
    return len(entry_schedule(days, i0, i1, cfg)[0])


# ---------------------------------------------------------------- per-trade and per-arm metrics
def trade_table(res: LabResult) -> pd.DataFrame:
    t = res.trades.copy()
    t["entry_date"] = pd.to_datetime(t.entry_ts).dt.date
    t["ror"] = t.pnl_per_contract / t.collateral_per_contract          # return on risk, per trade (sizing-free)
    t = t.merge(res.sizing[["trade_id", "contracts"]], on="trade_id", how="left") if res.sizing is not None else t
    t["contracts"] = t.get("contracts", pd.Series(1, index=t.index)).fillna(1)
    t["pnl_usd"] = t.pnl_per_contract * t.contracts
    return t


def max_drawdown(eq: pd.Series) -> float:
    if eq.empty:
        return float("nan")
    return float((eq / eq.cummax() - 1.0).min())


def arm_metrics(res: LabResult) -> dict:
    t = trade_table(res)
    eq = res.equity.equity if res.equity is not None else pd.Series(dtype=float)
    exits = t.exit_reason.value_counts().to_dict() if len(t) else {}
    out = {"trades": len(t)}
    if len(t):
        r = t.ror
        cyc = t.groupby(pd.to_datetime(t.entry_date).dt.to_period("M")).pnl_usd.sum()
        out.update(mean_ror=float(r.mean()), median_ror=float(r.median()), win_rate=float((t.pnl_per_contract > 0).mean()),
                   worst_trade_ror=float(r.min()), worst_5pct_ror=float(r.quantile(0.05)),
                   worst_1pct_ror=float(r.quantile(0.01)), worst_cycle=str(cyc.idxmin()), worst_cycle_usd=float(cyc.min()),
                   total_pnl_usd=float(t.pnl_usd.sum()))
    else:
        out.update(mean_ror=np.nan, median_ror=np.nan, win_rate=np.nan, worst_trade_ror=np.nan, worst_5pct_ror=np.nan,
                   worst_1pct_ror=np.nan, worst_cycle=None, worst_cycle_usd=np.nan, total_pnl_usd=0.0)
    out.update({f"exit_{k}": int(exits.get(k, 0)) for k in EXITS})
    out.update(max_dd=max_drawdown(eq), final_equity=float(eq.iloc[-1]) if len(eq) else np.nan)
    bl = res.blocked.reason.value_counts().to_dict() if len(res.blocked) else {}
    out["blocked"] = {k: int(v) for k, v in bl.items()}
    out["gap_days"] = int(sum(bl.get(k, 0) for k in GAP_REASONS) + res.meta["spot_coverage"]["days_without_spot"])
    out["leg_load_gaps"] = int(res.meta.get("leg_load_gaps", 0))
    return out


def by_year(res: LabResult) -> pd.DataFrame:
    t = trade_table(res)
    if not len(t):
        return pd.DataFrame(columns=["year", "trades", "win_rate", "mean_ror", "pnl_usd"])
    t["year"] = pd.to_datetime(t.entry_date).dt.year
    g = t.groupby("year")
    return pd.DataFrame({"trades": g.size(), "win_rate": g.pnl_per_contract.apply(lambda x: (x > 0).mean()),
                         "mean_ror": g.ror.mean(), "pnl_usd": g.pnl_usd.sum()}).reset_index()


# ---------------------------------------------------------------- what the guards skipped
def skipped_trades(m: Matched) -> pd.DataFrame:
    """Plain-arm trades whose entry day the guarded arm did not take, with the reason it gave."""
    pt = trade_table(m.plain)
    gt_dates = set(trade_table(m.guarded).entry_date)
    reason = {}
    for d, r in zip(m.guarded.blocked.date, m.guarded.blocked.reason):
        reason.setdefault(str(d), r)
    s = pt[~pt.entry_date.isin(gt_dates)].copy()
    s["guarded_reason"] = [reason.get(str(d), "unexplained") for d in s.entry_date]
    return s


def _skip_stats(s: pd.DataFrame) -> dict:
    if not len(s):
        return {"n": 0, "win_rate": np.nan, "mean_ror": np.nan, "total_usd": 0.0, "losers": 0, "winners": 0}
    return {"n": len(s), "win_rate": float((s.pnl_per_contract > 0).mean()), "mean_ror": float(s.ror.mean()),
            "total_usd": float(s.pnl_usd.sum()), "losers": int((s.pnl_per_contract <= 0).sum()),
            "winners": int((s.pnl_per_contract > 0).sum())}


def difference(m: Matched, skipped: pd.DataFrame) -> dict:
    p, g = arm_metrics(m.plain), arm_metrics(m.guarded)
    d = {k: g[k] - p[k] for k in ("trades", "mean_ror", "median_ror", "worst_5pct_ror", "worst_1pct_ror", "max_dd",
                                  "final_equity", "total_pnl_usd")}
    d["skipped"] = _skip_stats(skipped)
    d["skipped_by_reason"] = {r: _skip_stats(x) for r, x in skipped.groupby("guarded_reason")}
    return d


def split_sample(m: Matched, split: date) -> dict:
    p, g, s = trade_table(m.plain), trade_table(m.guarded), skipped_trades(m)
    out = {}
    for name, mask in (("first half", lambda x: x.entry_date < split), ("second half", lambda x: x.entry_date >= split)):
        out[name] = {"plain": _arm_slice(p[mask(p)]), "guarded": _arm_slice(g[mask(g)]), "skipped": _skip_stats(s[mask(s)])}
    return out


def _arm_slice(t: pd.DataFrame) -> dict:
    if not len(t):
        return {"trades": 0, "mean_ror": np.nan, "worst_5pct_ror": np.nan, "win_rate": np.nan, "total_pnl_usd": 0.0}
    return {"trades": len(t), "mean_ror": float(t.ror.mean()), "worst_5pct_ror": float(t.ror.quantile(0.05)),
            "win_rate": float((t.pnl_per_contract > 0).mean()), "total_pnl_usd": float(t.pnl_usd.sum())}


def switches(cfg: Config) -> dict:
    keys = ("gate", "gate_vix", "gate_day", "gate_5d", "upside_guard", "upside_5d", "event_buffer", "event_buffer_days",
            "event_types", "cb_days", "earnings_filter", "term_filter", "post_spike_wait", "min_ivr", "max_open_lots",
            "entry_requires_gate_trip", "force_entry", "delta_cap_pct", "tp", "stop_mult", "time_exit", "breach_exit",
            "fill", "budget_pct", "dte_target")
    d = cfg.to_dict()
    return {k: d[k] for k in keys}


# ---------------------------------------------------------------- output
def _f(x, pct=False, dp=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.{dp}f}%" if pct else f"{x:,.{dp}f}"


def _md(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render(m: Matched, split: date, source: str, events_note: str) -> str:
    p, g = arm_metrics(m.plain), arm_metrics(m.guarded)
    s = skipped_trades(m)
    diff = difference(m, s)
    L = [f"## {m.cadence.capitalize()} cadence (SPY iron condor, delta 0.16)", "",
         f"Data: {source} | events: {events_note} | scheduled entry days: {m.scheduled_days} | "
         f"switches that differ between arms: {', '.join(diff_switches(m.plain.cfg, m.guarded.cfg))}", ""]
    rows = [("trades", p["trades"], g["trades"]),
            ("entries blocked by a guard", sum(v for k, v in p["blocked"].items() if k in GUARD_REASONS),
             sum(v for k, v in g["blocked"].items() if k in GUARD_REASONS))]
    rows += [(f"blocked: {k}", p["blocked"].get(k, 0), g["blocked"].get(k, 0)) for k in sorted(set(p["blocked"]) | set(g["blocked"]))]
    rows += [("gap days (no quote / no spot)", p["gap_days"], g["gap_days"]),
             ("position-days not monitored (cache gap on a held leg)", p["leg_load_gaps"], g["leg_load_gaps"]),
             ("mean return on risk", _f(p["mean_ror"], True), _f(g["mean_ror"], True)),
             ("median return on risk", _f(p["median_ror"], True), _f(g["median_ror"], True)),
             ("win rate", _f(p["win_rate"], True, 1), _f(g["win_rate"], True, 1)),
             ("stops / TP / time / breach / expiry", "/".join(str(p[f"exit_{k}"]) for k in ("stop", "tp", "time", "breach", "expiry")),
              "/".join(str(g[f"exit_{k}"]) for k in ("stop", "tp", "time", "breach", "expiry"))),
             ("worst 5% trade (RoR)", _f(p["worst_5pct_ror"], True, 1), _f(g["worst_5pct_ror"], True, 1)),
             ("worst 1% trade (RoR)", _f(p["worst_1pct_ror"], True, 1), _f(g["worst_1pct_ror"], True, 1)),
             ("worst single trade (RoR)", _f(p["worst_trade_ror"], True, 1), _f(g["worst_trade_ror"], True, 1)),
             ("worst cycle (entry month, $)", f"{p['worst_cycle']} {_f(p['worst_cycle_usd'], dp=0)}",
              f"{g['worst_cycle']} {_f(g['worst_cycle_usd'], dp=0)}"),
             ("max drawdown of equity", _f(p["max_dd"], True, 1), _f(g["max_dd"], True, 1)),
             ("final equity ($)", _f(p["final_equity"], dp=0), _f(g["final_equity"], dp=0))]
    L += ["### Per arm", "", _md(["metric", "PLAIN", "GUARDED"], rows), ""]
    k = diff["skipped"]
    L += ["### Guarded minus plain", "", _md(["metric", "difference"], [
        ("trades", f"{diff['trades']:+d}"), ("mean RoR", _f(diff["mean_ror"], True)),
        ("worst 5% RoR", _f(diff["worst_5pct_ror"], True, 1)), ("max drawdown", _f(diff["max_dd"], True, 1)),
        ("final equity ($, includes sizing-path effect)", _f(diff["final_equity"], dp=0)),
        ("skipped trades: n / win rate / mean RoR", f"{k['n']} / {_f(k['win_rate'], True, 1)} / {_f(k['mean_ror'], True)}"),
        ("skipped trades: losers / winners", f"{k['losers']} / {k['winners']}"),
        ("skipped trades: total $ at plain sizing (negative = guards removed losses)", _f(k["total_usd"], dp=0))]), ""]
    rr = [(r, v["n"], _f(v["win_rate"], True, 1), _f(v["mean_ror"], True), _f(v["total_usd"], dp=0), v["losers"])
          for r, v in diff["skipped_by_reason"].items()]
    L += ["### Skipped trades by guard reason", "", _md(["guarded reason", "n", "win rate", "mean RoR", "total $", "losers"], rr), ""]
    yr = by_year(m.plain).merge(by_year(m.guarded), on="year", how="outer", suffixes=("_plain", "_guarded")).fillna(0)
    L += ["### Calendar year (entry year)", "", _md(["year", "trades plain", "trades guarded", "mean RoR plain",
          "mean RoR guarded", "$ plain", "$ guarded"],
          [(int(r.year), int(r.trades_plain), int(r.trades_guarded), _f(r.mean_ror_plain, True), _f(r.mean_ror_guarded, True),
            _f(r.pnl_usd_plain, dp=0), _f(r.pnl_usd_guarded, dp=0)) for r in yr.itertuples()]), ""]
    sp = split_sample(m, split)
    rows = []
    for half, v in sp.items():
        rows.append((half, v["plain"]["trades"], _f(v["plain"]["mean_ror"], True), _f(v["plain"]["worst_5pct_ror"], True, 1),
                     v["guarded"]["trades"], _f(v["guarded"]["mean_ror"], True), _f(v["guarded"]["worst_5pct_ror"], True, 1),
                     v["skipped"]["n"], _f(v["skipped"]["mean_ror"], True)))
    L += [f"### Split sample (split at {split})", "", _md(["half", "plain n", "plain mean RoR", "plain worst 5%",
          "guarded n", "guarded mean RoR", "guarded worst 5%", "skipped n", "skipped mean RoR"], rows), ""]
    return "\n".join(L)


def write_outputs(m: Matched, out: Path, split: date, source: str, events_note: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    trade_table(m.plain).to_csv(out / "trades_plain.csv", index=False)
    trade_table(m.guarded).to_csv(out / "trades_guarded.csv", index=False)
    skipped_trades(m).to_csv(out / "skipped_trades.csv", index=False)
    m.plain.blocked.to_csv(out / "blocked_plain.csv", index=False)
    m.guarded.blocked.to_csv(out / "blocked_guarded.csv", index=False)
    (out / "summary.md").write_text(render(m, split, source, events_note))
    js = {"cadence": m.cadence, "scheduled_days": m.scheduled_days, "plain": arm_metrics(m.plain),
          "guarded": arm_metrics(m.guarded), "difference": difference(m, skipped_trades(m)),
          "split": split_sample(m, split), "switches": {"plain": switches(m.plain.cfg), "guarded": switches(m.guarded.cfg)},
          "data_range": m.plain.meta["data_range"], "events": events_note, "source": source}
    (out / "summary.json").write_text(json.dumps(js, indent=2, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Matched guarded vs plain comparison (lab engine, cache only).")
    ap.add_argument("--provider", choices=["theta", "synthetic"], default="theta")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--events", default="data/events.csv")
    ap.add_argument("--start", default="2016-10-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--cadence", nargs="+", default=["monthly", "weekly"], choices=["monthly", "weekly"])
    ap.add_argument("--out", default="results/guard_matched")
    ap.add_argument("--syn-start", default="2015-01-02")
    ap.add_argument("--syn-end", default="2025-12-31")
    ap.add_argument("--syn-seed", type=int, default=7)
    a = ap.parse_args(argv)
    prov = factory.make(a.provider, "SPY", a.data_dir, a.syn_start, a.syn_end, a.syn_seed, rate=0.036)
    if prov.name == "synthetic":
        ev, note = prov.synthetic_events(), "SYNTHETIC-calendar(not real dates)"
    else:
        if not Path(a.events).exists():
            print(f"ERROR: events file not found: {a.events}. Build it with scripts/fetch_context_data.py (real dates only).",
                  file=sys.stderr)
            return 2
        ev, note = load_events(a.events), str(a.events)
    split = (pd.Timestamp(a.start) + (pd.Timestamp(a.end) - pd.Timestamp(a.start)) / 2).date()
    for cad in a.cadence:
        m = run_matched(prov, ev, note, cad, a.start, a.end)
        write_outputs(m, Path(a.out) / cad, split, prov.name, note)
        print(f"[{cad}] plain {len(m.plain.trades)} trades, guarded {len(m.guarded.trades)} | "
              f"skipped {len(skipped_trades(m))} -> {Path(a.out) / cad / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
