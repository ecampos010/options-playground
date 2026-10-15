"""Single-file, self-contained HTML report (inline CSS/JS/SVG; no external hosts)."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import CAVEAT, LEVEL_TAGS, START_EQUITY
from .montecarlo import CAVEAT_MC
from .replay import max_drawdown

ASSETS = Path(__file__).resolve().parent / "assets"
MAX_BYTES = 7_500_000
VIX_NAMES = {"lt15": "VIX < 15", "15to20": "VIX 15-20", "20to30": "VIX 20-30", "gt30": "VIX > 30"}


def e(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _f(x, nd=4):
    if x is None:
        return None
    try:
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return None
            if np.isinf(x):
                return 999.0
            return round(float(x), nd)
        return x
    except Exception:  # noqa: BLE001
        return None


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode("ascii")


def monthly_pack(eq: np.ndarray, days: pd.DatetimeIndex, start_eq: float):
    per = days.to_period("M")
    first = np.r_[True, per[1:] != per[:-1]]
    starts = np.where(first)[0]
    ends = np.r_[starts[1:], len(days)] - 1
    dd = max_drawdown(eq.astype(float), start_eq)
    ddm = np.maximum.reduceat(dd, starts)
    return eq[ends].astype(np.float32), ddm.astype(np.float32), [str(days[i].date()) for i in ends]


def table(headers, rows, num=(), cls="", tid=None, sortable=True) -> str:
    """rows: list of lists; cell = str or (text, sortvalue)."""
    h = "".join(f'<th scope="col" class="{"n" if i in num else ""}">{e(x)}</th>' for i, x in enumerate(headers))
    out = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            dv = ""
            if isinstance(c, tuple):
                c, v = c
                dv = f' data-v="{e(v)}"' if v is not None and not (isinstance(v, float) and np.isnan(v)) else ' data-v="nan"'
            raw = isinstance(c, str) and c.startswith("\0")
            txt = c[1:] if raw else e(c)
            tds.append(f'<td class="{"n" if i in num else ""}"{dv}>{txt}</td>')
        out.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="scroll"><table class="{"sortable " if sortable else ""}{cls}"' + (f' id="{tid}"' if tid else "") +
            f"><thead><tr>{h}</tr></thead><tbody>{''.join(out)}</tbody></table></div>")


def raw(s):
    return "\0" + s


def money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ("-", float("nan"))
    a = abs(x)
    s = f"${a / 1e6:,.2f}M" if a >= 1e6 else f"${a:,.0f}"
    return ("-" + s if x < 0 else s, float(x))


def pct(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ("-", float("nan"))
    return (f"{x * 100:.{d}f}%", float(x))


def build_rows(ok: pd.DataFrame, br: pd.DataFrame, series: dict, start_eq: float, mc_run: bool, horizon: int):
    rows, packs, axes = [], [], {}
    sidx = {}
    allr = pd.concat([ok.assign(_br=0), br.assign(_br=1)], ignore_index=True)
    for r in allr.to_dict("records"):
        sid = r["scenario_id"]
        s = series.get(sid)
        sref = None
        if s is not None:
            eqm, ddm, iso = monthly_pack(s["eq"], s["days"], start_eq)
            g = r["group"]
            axes.setdefault(g, iso)
            if len(axes[g]) == len(eqm):
                sref = len(packs)
                packs.append([_b64(eqm), _b64(ddm)])
        d = {
            "rk": int(r["rank"]), "lvl": int(r["level"]), "lt": r["level_tag"], "pf": r["profile"], "tg": r["config_tag"],
            "st": r["strategy"], "tk": r["tickers"], "nt": int(r["n_tickers"]), "A": int(r["allocation"]),
            "fe": _f(r["final_equity"], 0), "cagr": _f(r["cagr"]), "dd": _f(r["max_dd"]), "wm": _f(r["worst_month"]),
            "wt": _f(r["worst_trade_pct"]), "wu": _f(r["worst_trade_usd"], 0), "wr": _f(r["win_rate"], 3), "pfac": _f(r["profit_factor"], 2),
            "cal": None if r["_br"] else _f(r["calmar"], 3), "ac": _f(r["avg_contracts"], 1), "ls": int(r["longest_losing_streak"]),
            "br": int(r["_br"]), "bd": r["broke_date"], "yrs": _f(r["years"], 2), "n": int(r["n_trades"]),
            "few": int(bool(r["too_few_trades"])), "zn": r["zone"], "ruin": _f(r.get("ruin_odds")), "dl": _f(r.get("short_delta"), 3),
            "vx": r.get("typical_vix", ""), "v": r["verdict"], "s": sref,
            "hm": int(mc_run), "p10": _f(r.get(f"mc{horizon}_final_p10"), 0), "p50": _f(r.get(f"mc{horizon}_final_p50"), 0),
            "p90": _f(r.get(f"mc{horizon}_final_p90"), 0),
        }
        rows.append(d)
    return rows, packs, axes


def summary_tiles(meta, ok, br, df) -> str:
    z = df["zone"].value_counts()
    t = [("Scenarios replayed", f"{len(df):,}"), ("Survived (not broke)", f"{len(ok):,}"), ("Went broke", f"{len(br):,}"),
         ("Trades in logs", f"{meta['n_trades']:,}"), ("Safe / Mild", f"{int(z.get('Safe', 0))} / {int(z.get('Mild', 0))}"),
         ("Danger / Breaking point", f"{int(z.get('Danger', 0))} / {int(z.get('Breaking point', 0))}")]
    return '<div class="tiles">' + "".join(f'<div class="tile"><span class="muted small">{e(a)}</span><b>{e(b)}</b></div>' for a, b in t) + "</div>"


def build_html(ctx: dict) -> str:
    meta = ctx["meta"]
    ok, br, df = ctx["ok"], ctx["br"], ctx["df"]
    mc_run = meta["mc_run"]
    horizon = ctx["thermo_cfg"]["ruin_horizon_months"]
    rows, packs, axes = build_rows(ok, br, ctx["series"], START_EQUITY, mc_run, horizon)
    syn = meta["synthetic"]
    zones = [{"name": z["name"], "ruin": z["ruin"], "dd": z["dd"]} for z in ctx["thermo_cfg"]["zones"]]
    zone_names = [z["name"] for z in zones] + [ctx["thermo_cfg"].get("fallback_zone", "Breaking point")]
    spy = meta.get("spy")
    data = {
        "meta": {"synthetic": syn, "start": START_EQUITY, "apy": meta["apy"], "spy": spy}, "rows": rows, "series": packs, "axes": axes,
        "zones": zones, "zone_names": zone_names, "vixnames": VIX_NAMES, "horizon": horizon, "mixname": ctx["mixname"],
    }

    def render(data):
        return _page(ctx, data, syn)

    page = render(data)
    if len(page.encode()) > MAX_BYTES:  # drop chart series from the lowest-ranked rows until it fits
        order = [i for i, r in sorted(enumerate(rows), key=lambda t: (t[1]["br"] == 0, t[1]["rk"]), reverse=True) if r["s"] is not None]
        k = 0
        while len(page.encode()) > MAX_BYTES and k < len(order):
            for i in order[k:k + 250]:
                rows[i]["s"] = None
            k += 250
            page = render(data)
    return page


def _page(ctx, data, syn) -> str:
    meta = ctx["meta"]
    horizon = data["horizon"]
    css = (ASSETS / "report.css").read_text()
    js = (ASSETS / "report.js").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    title = ("SYNTHETIC " if syn else "") + "Options playground report"
    parts = []
    parts.append(f'<h1>{e(title)}</h1><div class="caveat">{e(CAVEAT)}</div>')
    if syn:
        parts.append('<div class="synth">SYNTHETIC: every number on this page comes from generated fixture logs (make-synthetic-logs), '
                     "not from market data. It only proves the machinery works.</div>")
    parts.append(f'<p class="muted small">Generated {e(meta["generated"])} from {e(meta["runs_desc"])}. Research software, not advice. '
                 f'Start equity $100,000; idle cash earns {meta["apy"] * 100:.1f}% APY; sizing = allocation x current equity split equally across tickers; '
                 f'per-expiry cap {e(meta["expiry_cap"])}; circuit breaker {e(meta["cb"])}. Monte Carlo: {e(meta["mc_desc"])}. Delta source: {e(meta.get("delta_source", "unknown"))}.</p>')
    parts.append('<nav class="toc"><a href="#explorer">Equity &amp; thermometer</a><a href="#ranking">Ranking</a><a href="#vix">VIX regimes</a>'
                 '<a href="#ways">Ways to lose money</a><a href="#stress">Stress lab</a><a href="#mgmt">Management rules</a><a href="#method">Caveats</a>'
                 '<button id="btn-theme" type="button" aria-label="Toggle light/dark">Light / dark</button></nav>')
    parts.append(summary_tiles(meta, ctx["ok"], ctx["br"], ctx["df"]))
    # explorer
    parts.append('<h2 id="explorer">Equity by allocation, drawdown and thermometer</h2><div class="card">'
                 '<div class="controls"><label>Strategy<select id="sel-strat"></select></label><label>Level / profile / config<select id="sel-var"></select></label>'
                 '<label>Tickers<select id="sel-tk"></select></label><label>Scenario for drawdown &amp; gauge<select id="sel-al"></select></label>'
                 '<button id="btn-log" type="button" aria-pressed="false">Log scale</button></div>'
                 '<h3 id="eq-title"></h3><div class="chart" id="eq-chart"></div><div class="legend" id="eq-legend"></div>'
                 '<p class="small muted" id="eq-note"></p></div>'
                 '<div class="card"><h3 id="dd-title"></h3><div class="chart" id="dd-chart"></div></div>'
                 '<div class="card"><h3>Thermometer</h3><div id="thermo"></div></div>')
    # ranking
    parts.append('<h2 id="ranking">Ranking</h2><p class="small">Ranked within thermometer zone first (Safe, Mild, Danger, Breaking point), then by return / max drawdown (CAGR divided by max drawdown), among scenarios that did '
                 '<b>not</b> go broke; final equity is shown beside it and is never the ranking key. Scenarios with fewer than 30 trades are listed after the '
                 'rest and flagged TOO FEW TRADES - not evidence. Broke scenarios are a separate list, ranked by how late they went broke. '
                 'Level tags and zone names carry the meaning; colour is decoration only. Click a row to load it above.</p>'
                 '<div class="controls"><label>List<select id="f-view"><option value="ok">Survivors (ranked)</option><option value="br">Went broke (latest first)</option></select></label>'
                 '<label>Level<select id="f-lvl"></select></label><label>Strategy<select id="f-st"></select></label><label># tickers<select id="f-nt"></select></label>'
                 '<label>Zone<select id="f-zn"></select></label><label>Search<input id="f-q" type="search" placeholder="ticker, profile, text"></label></div>'
                 '<p class="small muted" id="rank-count"></p><div class="scroll"><table><thead id="rank-head"></thead><tbody id="rank-body"></tbody></table></div>'
                 '<p><button id="more" type="button">Show more</button></p>')
    # vix
    v = ctx["vix"]
    vr = []
    for r in v.to_dict("records"):
        vr.append([VIX_NAMES.get(r["bucket"], r["bucket"]) + (" - thin evidence" if r["thin_evidence"] else ""), (f"{r['trades']:,}", r["trades"]),
                   pct(r["win_rate"]), pct(r["avg_pnl_pct_collateral"], 2), pct(r["worst_pnl_pct_collateral"], 1),
                   (f"{r['max_alloc_safe_or_mild']}%" if r["max_alloc_safe_or_mild"] else "none", r["max_alloc_safe_or_mild"] or 0),
                   (f"{r['widest_delta_safe_or_mild']:.2f}" if r["widest_delta_safe_or_mild"] is not None and not pd.isna(r["widest_delta_safe_or_mild"]) else "none", r["widest_delta_safe_or_mild"] if r["widest_delta_safe_or_mild"] is not None else float("nan")),
                   r["where"]])
    parts.append('<h2 id="vix">VIX-regime lookup</h2><p class="small">All trades grouped by VIX bucket at entry. "Highest allocation / widest delta" come from replays that '
                 "only enter when VIX was in that bucket (all tickers of the log) and stayed in Safe or Mild. Buckets under 30 trades are marked thin evidence.</p>" +
                 table(["Regime", "Trades", "Win rate", "Avg P&L % collateral", "Worst P&L % collateral", "Highest alloc Safe/Mild", "Widest short delta Safe/Mild", "Where"], vr, num=(1, 2, 3, 4, 5, 6)))
    # ways
    w = ctx["ways"]
    wr = []
    for r in w.to_dict("records"):
        if r.get("status") != "ok":
            wr.append([r["mistake"], r.get("how", ""), "not run", "not run", "not run", "not run", "not run"])
        else:
            wr.append([r["mistake"], r["how"], pct(r["p_broke"], 2), pct(r["p_dd50"], 1), (f"{r['multiple']:.1f}x", r["multiple"]), money(r["median_final"]), pct(r["replay_max_dd"], 0)])
    an = ctx["analytic"]
    ar = [[f"{int(r['allocation'])}%", (f"{r['n_exact']:.2f}", r["n_exact"]), (str(int(r["n_rounded"])), r["n_rounded"]), (str(int(r["n_to_reach"])), r["n_to_reach"])] for r in an.to_dict("records")]
    parts.append('<h2 id="ways">How to become an expert at losing money</h2><p class="small">' + e(CAVEAT_MC) + f" P(broke) is over {horizon} months. Multiple = P(broke) / base P(broke), "
                 "base floored at 0.1%; ties broken by P(drawdown > 50%). Missing logs show as <b>not run</b>.</p>" +
                 table(["Mistake", "How", f"P(broke) {horizon}m", "P(DD>50%)", "x base", "Median final", "Replay max DD"], wr, num=(2, 3, 4, 5, 6)) +
                 '<h3>Analytic: consecutive full losses to lose 90%</h3><p class="small">n = ln(0.10) / ln(1 - A). <b>No stop and no credit: the worst case.</b></p>' +
                 table(["Allocation", "n (exact)", "n (rounded)", "Losses needed to reach -90%"], ar, num=(1, 2, 3)))
    # stress
    st = ctx["stress"]
    sr = [[r["kind"], r["window"], r["config_tag"], r["strategy"], (str(r["trades"]), r["trades"]), money(r["pnl_mean"]), money(r["pnl_worst"]), money(r["worst_mark"]), r["exit_reasons"], r["slippage_through_stop"]] for r in st.to_dict("records")] if len(st) else []
    bp = ctx["bp"]
    parts.append('<h2 id="stress">Stress lab</h2><p class="small">P&amp;L is per contract. Slippage through stop = a stop exit whose gross loss exceeds the stop level by more than 2%.</p>' +
                 (table(["Kind", "Window / profile", "Config", "Strategy", "Trades", "Mean P&L", "Worst P&L", "Worst mark", "Exit reasons", "Slippage through stop"], sr, num=(4, 5, 6, 7)) if sr else "<p>No stress_* or experiment_* logs found: <b>not run</b>.</p>") + bp["html"])
    # mgmt
    mg = ctx["mgmt"]
    mr = []
    for r in mg["table"].to_dict("records") if len(mg["table"]) else []:
        mr.append([f"{r['profile']} / {r['config_tag']}", (str(r["early_exits"]), r["early_exits"]), money(r["saved_usd"]), money(r["cost_usd"]), money(r["net_usd"]), pct(r["whipsaw_rate"], 0),
                   (str(r["matched_entries"]), r["matched_entries"]), money(r["vs_base_mean"]), money(r["first_half"]), money(r["second_half"]), r["label"]])
    parts.append(f'<h2 id="mgmt">Management-rule scorecard</h2><p class="small"><b>{mg["n_configs"]} configurations compared</b> against baseline '
                 f'<code>{e(mg["baseline"])}</code>. Saved = loss avoided vs holding to expiry; cost = gain given up; whipsaw = share of protective exits (stop, breach, trigger) '
                 f'that would have finished profitable. "vs baseline" is the mean per-contract P&amp;L difference on the same entries (same ticker, strategy, entry date). '
                 f'A rule is "holds up" only if it beats baseline in both the first and second half of the sample; otherwise "unproven". '
                 f'Configs with different strikes (delta/profile variants) are not management rules and are excluded: {e(", ".join(mg.get("excluded_other_strikes", [])) or "none")}.</p>' +
                 (table(["Rule", "Early exits", "Saved", "Cost", "Net vs holding", "Whipsaw", "Matched entries", "Mean vs baseline / contract", "1st half", "2nd half", "Label"], mr, num=(1, 2, 3, 4, 5, 6, 7, 8, 9)) if mr else "<p>not run (no baseline config found).</p>"))
    parts.append('<h2 id="method">What this does NOT tell you</h2><ul class="small"><li>A ~10-year sample: relative comparison only, not a forecast; tails are understated.</li>'
                 "<li>Bootstrap resampling can never create a month worse than the worst one observed.</li><li>No taxes, no margin-call mechanics, no early-assignment modelling beyond the logged flag.</li>"
                 "<li>Unit logs are replayed at fixed sizing rules; real fills at size would differ (liquidity, slippage).</li>"
                 f"<li>{e(CAVEAT)}</li></ul>")
    body = "\n".join(parts)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta name="color-scheme" content="light dark"><title>{e(title)}</title><style>{css}</style></head><body><div class="wrap">{body}</div>'
            f'<script id="data" type="application/json">{payload}</script><script>{js}</script></body></html>')
