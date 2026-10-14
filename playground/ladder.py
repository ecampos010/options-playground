"""Matrix grid and the traffic-light ladder (Green / Yellow E1..E7 / Red)."""
from __future__ import annotations

from .config import Config, apply_overrides, build_config, diff_switches
from .engine import run_backtest
from .report import metrics, table

MATRIX_GUARDS = [
    ("gate off", dict(gate=False, upside_guard=False, cb_days=0)),
    ("gate on", dict(gate=True, upside_guard=False, cb_days=0)),
    ("gate+upside", dict(gate=True, upside_guard=True, cb_days=0)),
    ("gate+upside+cb21", dict(gate=True, upside_guard=True, cb_days=21)),
]
MATRIX_EXITS = [("A tp75 noTime", dict(tp=0.75, time_exit=0)), ("B tp50+21DTE", dict(tp=0.50, time_exit=21))]
NA_PROFILES = {"yellow_E5": "N/A for SPY", "yellow_E6": "N/A single-underlying"}
LADDER_PROFILES = ["green"] + [f"yellow_E{i}" for i in range(1, 8)] + ["red"]


def matrix_configs(base: Config):
    for gname, g in MATRIX_GUARDS:
        for ename, e in MATRIX_EXITS:
            yield f"{gname} / {ename}", apply_overrides(base, {**g, **e})


def run_matrix(provider, base: Config, events, note: str, fill_compare: bool = False) -> str:
    fills = ["natural", "mid"] if fill_compare else [base.fill]
    rows = []
    for name, cfg in matrix_configs(base):
        cells = []
        for f in fills:
            m = metrics(run_backtest(provider, apply_overrides(cfg, {"fill": f}), events, note))
            cells += [f"{m['final']:,.0f}", f"{m['maxdd']:.1%}", m["trades"], f"{m['win']:.0%}"]
        rows.append([name] + cells)
    head = ["variant"]
    for f in fills:
        head += [f"final[{f}]", f"maxDD[{f}]", f"n[{f}]", f"win[{f}]"]
    return (f"MATRIX | data={provider.name} | fill={'+'.join(fills)} | seed={base.seed} | budget={base.effective_budget():.0%} | "
            f"entry={base.entry_window}/{base.entry_sample} | res={base.resolution} | events={note}\n"
            + table(rows, head))


def run_ladder(provider, overrides: dict, events, note: str, equal_budget: bool = False,
               split_sample: bool = False) -> str:
    base = build_config("green", overrides)
    days = provider.trading_days()
    lo = base.start_date or str(days[6])
    hi = base.end_date or str(days[-1])
    ranges = [("full", lo, hi)]
    if split_sample:
        mid = str(days[(days.index(next(d for d in days if str(d) >= lo)) + days.index(
            max(d for d in days if str(d) <= hi))) // 2])
        ranges = [("first half", lo, mid), ("second half", mid, hi)]
    out = []
    for label, a, b in ranges:
        ov = {**overrides, "start_date": a, "end_date": b}
        rows, ref, years = [], None, 0.0
        for name in LADDER_PROFILES:
            ovp = dict(ov)
            cfg = build_config(name, ovp)
            if equal_budget and "budget_pct" not in overrides:
                cfg = apply_overrides(cfg, {"budget_pct": build_config("green", ovp).effective_budget()})
            m = metrics(run_backtest(provider, cfg, events, note))
            years = max(years, m["years"])
            ref = m if name == "green" else ref
            diff = m["final"] - ref["final"]
            flag = ""
            if m["trades"] < 10:
                flag = "TOO FEW TRADES - not evidence"
            elif name in NA_PROFILES:
                flag = NA_PROFILES[name]
            rows.append([name, m["trades"], f"{m['win']:.0%}" if m["trades"] else "-",
                         f"{m['avg_credit']:.2f}" if m["trades"] else "-", f"{m['final']:,.0f}",
                         f"{m['cagr']:.1%}", f"{m['maxdd']:.1%}",
                         "(ref)" if name == "green" else f"{diff:+,.0f}", flag])
        out.append(f"LADDER [{label} {a}..{b}] | data={provider.name} | fill={base.fill} | seed={base.seed} | "
                   f"entry={base.entry_window}/{base.entry_sample} | res={base.resolution} | events={note}"
                   f"{' | equal-budget' if equal_budget else ' | tier budgets 15/5/1%'}\n"
                   + table(rows, ["profile", "trades", "win", "avg credit", "final equity", "CAGR", "max DD",
                                  "vs green", "flag"]))
        if provider.uses_real_data:
            out.append(f"Real-quote evidence covers {years:.0f} years; the project's promotion rule asks for 20+ "
                       f"years plus a split-sample check, so results here are a 10-year tier only.")
        else:
            out.append(f"SYNTHETIC data: zero years of real-quote evidence. (Real-quote evidence will cover N<=10 years; "
                       f"the project's promotion rule asks for 20+ years plus a split-sample check, so real results "
                       f"here are a 10-year tier only.)")
    return "\n".join(out)


def switch_report() -> dict[str, list[str]]:
    g = build_config("green")
    return {n: diff_switches(g, build_config(n)) for n in LADDER_PROFILES}
