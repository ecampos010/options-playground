"""`report` command: logs -> grid -> ranking -> ways/stress/mgmt -> outputs."""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import ALLOCATIONS, CAVEAT, LEVEL_TAGS, START_EQUITY
from . import grid as G
from . import htmlreport, mgmt, stress, thermo, verdicts, ways
from .replay import ReplayConfig, build_variant
from .schema import load_runs


@dataclass
class ReportOptions:
    allocations: list = field(default_factory=lambda: list(ALLOCATIONS))
    apy: float = 0.036
    cb: int = 0
    expiry_cap: float | None = 0.60
    paths: int = 10000
    block: int = 2
    seed: int = 7
    no_mc: bool = False
    jobs: int = 1
    spy_csv: str | None = None
    bp_strategy: str = "iron_condor"
    bp_tickers: list = field(default_factory=lambda: ["SPY"])
    bp_profile: str = "^green"
    bp_dd: float = 0.5
    baseline_tag: str | None = None
    config_dir: str | None = None
    synthetic: bool | None = None


def read_spy(path: str | None):
    if not path:
        return None, "SPY buy-and-hold line omitted: no --spy-csv supplied."
    p = Path(path)
    if not p.exists():
        return None, f"SPY buy-and-hold line omitted: {path} not found."
    d = pd.read_csv(p)
    low = {c.lower().replace(" ", "_"): c for c in d.columns}
    dc = low.get("date") or d.columns[0]
    pc = next((low[k] for k in ("adj_close", "close", "adjclose") if k in low), None)
    if pc is None:
        return None, f"SPY buy-and-hold line omitted: no close column in {path}."
    d = d[[dc, pc]].dropna()
    d[dc] = pd.to_datetime(d[dc])
    m = d.set_index(dc)[pc].astype(float).sort_index().resample("ME").last().dropna()
    return {"t": [int(x.value // 10**6) for x in m.index], "v": [round(float(v), 4) for v in m.values]}, "SPY reference line from " + p.name


def bp_section(trades, marks, rcfg, o: ReportOptions, gcfg):
    sel = trades[trades["profile"].str.contains(o.bp_profile, case=False, regex=True) & (trades["strategy"] == o.bp_strategy)]
    if sel.empty or not set(o.bp_tickers) <= set(sel["ticker"]):
        return {"found": None, "html": "<h3>Breaking-point finder</h3><p>not run (no matching log for the chosen strategy/tickers).</p>", "md": "## Breaking-point finder\n\nnot run (no matching log).\n", "line": "breaking point: not run"}
    tag = "baseline" if "baseline" in set(sel["config_tag"]) else sel["config_tag"].value_counts().index[0]
    sub = sel[sel["config_tag"] == tag]
    vd = build_variant(sub, marks[marks["uid"].isin(set(sub["uid"]))])
    mask = np.isin(vd.trades["ticker"].to_numpy(), o.bp_tickers)
    r = stress.breaking_point(vd, mask, rcfg, dd_limit=o.bp_dd)
    name = f"{sub['profile'].iloc[0]} / {tag} / {o.bp_strategy} / {'+'.join(o.bp_tickers)}"
    L = [f"## Breaking-point finder", "", f"Setup: **{name}**. Allocation raised in 1% steps until max drawdown > {int(o.bp_dd * 100)}% or the account goes broke.", ""]
    H = [f"<h3>Breaking-point finder</h3><p class='small'>Setup: <b>{htmlreport.e(name)}</b>. Allocation raised in 1% steps until max drawdown &gt; {int(o.bp_dd * 100)}% or the account goes broke.</p>"]
    if r["found"]:
        line = f"breaking point: {r['allocation']}% allocation ({r['kind']}) on {r['date']}"
        L.append(f"**First failing allocation: {r['allocation']}%** ({r['kind']}), first hit on {r['date']} (max drawdown at that allocation {r['max_dd_at_alloc']:.0%}).")
        H.append(f"<p><b>First failing allocation: {r['allocation']}%</b> ({htmlreport.e(r['kind'])}), first hit on {r['date']}; max drawdown at that allocation {r['max_dd_at_alloc']:.0%}.</p>")
        if r["causes"]:
            L += ["", "Trades behind it (worst realised losses in the 45 days before):", "", "| trade | ticker | strategy | exit | reason | contracts | P&L |", "|---|---|---|---|---|---|---|"]
            rows = []
            for c in r["causes"]:
                L.append(f"| {c['trade_id']} | {c['ticker']} | {c['strategy']} | {c['exit_ts'][:10]} | {c['exit_reason']} | {c['contracts']} | ${c['usd']:,.0f} |")
                rows.append([c["trade_id"], c["ticker"], c["strategy"], c["exit_ts"][:10], c["exit_reason"], str(c["contracts"]), htmlreport.money(c["usd"])])
            H.append(htmlreport.table(["Trade", "Ticker", "Strategy", "Exit", "Reason", "Contracts", "P&L"], rows, num=(5, 6)))
        else:
            L.append("\nNo single realised trade dominates: the failure came from open-position marks.")
            H.append("<p>No single realised trade dominates: the failure came from open-position marks.</p>")
    else:
        line = f"breaking point: none up to {r['a_max']}%"
        L.append(f"No failure up to {r['a_max']}% allocation.")
        H.append(f"<p>No failure up to {r['a_max']}% allocation.</p>")
    tab, hit = stress.first_delta_over(trades, marks, o.bp_strategy, o.bp_tickers, alloc=15, paths=gcfg.paths, block=gcfg.block, seed=gcfg.seed, cfg=rcfg) if not o.no_mc else (pd.DataFrame(), None)
    if len(tab):
        L += ["", "### Delta ladder (15% allocation, 36-month Monte Carlo)", "", "| profile / config | short delta | P(broke) | replay max DD |", "|---|---|---|---|"]
        rows = []
        for t in tab.to_dict("records"):
            L.append(f"| {t['profile']} / {t['config_tag']} | {t['short_delta']:.2f} | {t['p_broke']:.2%} | {t['max_dd']:.0%} |")
            rows.append([f"{t['profile']} / {t['config_tag']}", (f"{t['short_delta']:.2f}", t["short_delta"]), htmlreport.pct(t["p_broke"], 2), htmlreport.pct(t["max_dd"], 0)])
        msg = (f"First delta setting with P(broke) above 5%: **{hit['short_delta']:.2f}** ({hit['profile']} / {hit['config_tag']}, {hit['p_broke']:.1%})."
               if hit else "No delta setting in the logs pushed P(broke) above 5% at a 15% allocation.")
        L += ["", msg]
        H.append("<h4>Delta ladder (15% allocation, 36-month Monte Carlo)</h4>" + htmlreport.table(["Profile / config", "Short delta", "P(broke)", "Replay max DD"], rows, num=(1, 2, 3)) + f"<p>{htmlreport.e(msg.replace('**', ''))}</p>")
        line += "; " + (f"first delta with P(broke)>5%: {hit['short_delta']:.2f}" if hit else "no delta >5% P(broke)")
    elif not o.no_mc:
        L += ["", "Delta ladder: not run (fewer than one delta variant with the chosen tickers)."]
        H.append("<p>Delta ladder: not run (no delta variants in the logs).</p>")
    return {"found": r, "html": "".join(H), "md": "\n".join(L) + "\n", "line": line}


def stress_md(st: pd.DataFrame, bp_md: str, synthetic: bool) -> str:
    L = ["# Stress lab", "", f"> {CAVEAT}"] + (["> **SYNTHETIC** - generated fixtures, not market data."] if synthetic else []) + [""]
    if len(st):
        L += ["| kind | window / profile | config | strategy | trades | mean P&L | worst P&L | worst mark | exit reasons | slippage through stop |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in st.to_dict("records"):
            L.append(f"| {r['kind']} | {r['window']} | {r['config_tag']} | {r['strategy']} | {r['trades']} | ${r['pnl_mean']:,.0f} | ${r['pnl_worst']:,.0f} | ${r['worst_mark']:,.0f} | {r['exit_reasons']} | {r['slippage_through_stop']} |")
    else:
        L.append("No stress_* / experiment_* logs: not run.")
    return "\n".join(L) + "\n\n" + bp_md


def verdicts_md(ok, br, synthetic, delta_source: str = "unknown") -> str:
    L = ["# Verdicts", "", f"> {CAVEAT}"] + (["> **SYNTHETIC** - generated fixtures, not market data."] if synthetic else []) + [
        "", f"Delta source in these logs: **{delta_source}** (theta_greeks = ThetaData first_order Greeks; bs_fallback = Black-Scholes from mid IV).",
        "", "Ranked within thermometer zone first (Safe, Mild, Danger, Breaking point), then by return / max drawdown, among scenarios that did not go broke (final equity is never the key).", "", "## Survivors", ""]
    for r in ok.to_dict("records"):
        L.append(f"{r['rank']}. **[{r['level_tag']}]** {r['strategy']} {r['tickers']} @ {int(r['allocation'])}% ({r['profile']}/{r['config_tag']}) - zone {r['zone']}; "
                 f"ret/DD {r['calmar']:.2f}, final ${r['final_equity']:,.0f}. {r['verdict']}")
    L += ["", "## Went broke (latest first)", ""]
    for r in br.to_dict("records"):
        L.append(f"{r['rank']}. **[{r['level_tag']}]** {r['strategy']} {r['tickers']} @ {int(r['allocation'])}% ({r['profile']}/{r['config_tag']}) - {r['verdict']}")
    return "\n".join(L) + "\n"


def run_report(runs: str, out: str, o: ReportOptions, log=print) -> dict:
    t0 = time.time()
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    lg = load_runs(runs)
    errs = [p for p in lg.problems if p[0] == "error"]
    if errs:
        for _, r, m in errs[:20]:
            log(f"[ERROR] {r}: {m}")
        raise SystemExit(f"schema validation failed ({len(errs)} error(s)); run `python -m reporting validate {runs}`")
    synthetic = lg.synthetic if o.synthetic is None else o.synthetic
    cdir = Path(o.config_dir) if o.config_dir else thermo.CONFIG_DIR
    tcfg = thermo.load_config(cdir / "thermometer.yaml")
    vcfg = verdicts.load_config(cdir / "verdicts.yaml")
    wcfg = ways.load_config(cdir / "ways_to_lose.yaml")
    wcfg["base_floor"] = vcfg.get("ways_to_lose", {}).get("base_floor", 0.001)
    rcfg = ReplayConfig(apy=o.apy, expiry_cap=o.expiry_cap, cb_days=o.cb)
    gcfg = G.GridConfig(allocations=o.allocations, replay=rcfg, mc=not o.no_mc, paths=o.paths, block=o.block, seed=o.seed,
                        jobs=o.jobs, thermo_cfg=tcfg)
    log(f"loaded {len(lg.trades):,} trades / {len(lg.marks):,} marks from {runs}{'  [SYNTHETIC]' if synthetic else ''}")
    df, series = G.run_grid(lg.trades, lg.marks, gcfg, progress=None)
    if df.empty:
        raise SystemExit("no scenarios could be built (only stress logs?)")
    log(f"grid: {len(df):,} scenarios in {time.time() - t0:.0f}s")
    df = G.add_verdicts(df, vcfg)
    ok, br = G.rank(df)
    allr = pd.concat([ok.assign(list="ranked"), br.assign(list="broke")], ignore_index=True)
    res = allr.drop(columns=["group"]).copy()
    res["synthetic"] = synthetic
    res.to_csv(outp / "results.csv", index=False)
    (outp / "verdicts.md").write_text(verdicts_md(ok, br, synthetic, lg.delta_source_summary()))
    ids = G.select_for_parquet(ok, br)
    G.write_parquet(outp, ids, series)
    # ways to lose
    w = ways.Ways(lg.trades, lg.marks, wcfg, rcfg, paths=o.paths, block=o.block, seed=o.seed, horizon=tcfg["ruin_horizon_months"])
    wtab = w.table() if not o.no_mc else pd.DataFrame([{"mistake": "(Monte Carlo disabled)", "how": "--no-mc", "status": "not run"}])
    an = ways.consecutive_losses_table()
    (outp / "ways_to_lose_money.md").write_text(ways.ways_markdown(wtab, an, tcfg["ruin_horizon_months"], synthetic))
    # stress
    st = stress.stress_table(lg.trades, lg.marks, lg.configs)
    bp = bp_section(lg.trades, lg.marks, rcfg, o, gcfg)
    (outp / "stress.md").write_text(stress_md(st, bp["md"], synthetic))
    mg = mgmt.scorecard(lg.trades, o.baseline_tag)
    vix = G.vix_regime_lookup(lg.trades, lg.marks, gcfg)
    spy, spy_msg = read_spy(o.spy_csv)
    nm = len(set(zip(lg.trades["profile"], lg.trades["config_tag"])))
    meta = {"synthetic": synthetic, "apy": o.apy, "spy": spy, "n_trades": len(lg.trades), "mc_run": not o.no_mc,
            "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "runs_desc": f"{len(lg.configs)} run(s), {nm} profile/config variants in {runs}",
            "delta_source": lg.delta_source_summary(),
            "expiry_cap": "off" if o.expiry_cap is None else f"{o.expiry_cap:.0%} of a ticker's share per expiry", "cb": "off" if not o.cb else f"{o.cb}-day pause after a losing month",
            "mc_desc": "not run" if o.no_mc else f"{o.paths:,} paths, {o.block}-month blocks, seed {o.seed}, horizons 12 and 36 months; " + "bootstrap understates tails"}
    ctx = {"meta": meta, "ok": ok, "br": br, "df": df, "series": series, "thermo_cfg": tcfg, "vix": vix, "ways": wtab, "analytic": an,
           "stress": st, "bp": bp, "mgmt": mg, "mixname": G.MIX}
    page = htmlreport.build_html(ctx)
    (outp / "index.html").write_text(page, encoding="utf-8")
    # summary
    S = []
    S.append(f"{'SYNTHETIC ' if synthetic else ''}report -> {outp}/index.html ({len(page) / 1e6:.2f} MB)  | {CAVEAT}")
    S.append(f"scenarios {len(df):,}: survived {len(ok):,}, went broke {len(br):,}; zones " + ", ".join(f"{k} {v}" for k, v in df['zone'].value_counts().items()))
    S.append(f"delta source: {lg.delta_source_summary()}")
    S.append(f"parquet equity files: {len(ids)}  | {spy_msg}")
    S.append("top 5 by return/maxDD (not broke):")
    for r in ok[~ok["too_few_trades"]].head(5).to_dict("records"):
        S.append(f"  {r['rank']:>3}. [{r['level_tag']}] {r['strategy']} {r['tickers']} @{int(r['allocation'])}% ret/DD {r['calmar']:.2f} maxDD {r['max_dd']:.0%} final ${r['final_equity']:,.0f} ({r['zone']})")
    if len(br):
        r = br.iloc[0]
        S.append(f"latest-broke: {r['strategy']} {r['tickers']} @{int(r['allocation'])}% on {r['broke_date']}")
    okw = wtab[wtab.get("status", "") == "ok"] if "status" in wtab else wtab
    S.append("ways to lose (P(broke) 36m, x base): " + "; ".join(f"{r['mistake']} {r['p_broke']:.1%} ({r['multiple']:.1f}x)" for r in okw.iloc[1:4].to_dict("records")) if len(okw) > 1 else "ways to lose: base not run")
    nr = int((wtab.get("status", pd.Series(dtype=str)) != "ok").sum()) if "status" in wtab else 0
    S.append(f"  ({nr} row(s) 'not run')")
    S.append(bp["line"])
    S.append(f"stress rows {len(st)}; mgmt scorecard: {mg['n_configs']} configs vs baseline '{mg['baseline']}' - " + (", ".join(f"{k} {v}" for k, v in mg["table"]["label"].value_counts().items()) if len(mg["table"]) else "none"))
    S.append(f"done in {time.time() - t0:.0f}s")
    for s in S:
        log(s)
    return {"df": df, "ok": ok, "br": br, "summary": S, "ctx": ctx, "page": page}
