from __future__ import annotations

import argparse
import os
import sys


def _alloc(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m reporting", description="Reporting, portfolio replay and stress lab (research software, not advice).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate a run folder, or a folder of runs, against the trade-log schema")
    v.add_argument("runs", help="a single run folder (contains trades.csv) or a parent folder of run folders")

    g = sub.add_parser("make-synthetic-logs", help="write SYNTHETIC fixture logs")
    g.add_argument("--out", default="runs_synth")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--start", default="2016-10-03")
    g.add_argument("--end", default="2026-09-30")
    g.add_argument("--cadence", type=int, default=5, help="entry cadence in trading days")

    def add_report_flags(p):
        p.add_argument("--allocations", type=_alloc, default=[5, 15, 20, 50, 100], help="comma list of percents")
        p.add_argument("--apy", type=float, default=0.036)
        p.add_argument("--cb", type=int, default=0, choices=[0, 21], help="replay circuit breaker: 0 off, 21 trading-day pause after a losing month")
        p.add_argument("--expiry-cap", type=float, default=0.60, help="per-expiry cap as share of a ticker's budget (0 = off)")
        p.add_argument("--paths", type=int, default=10000)
        p.add_argument("--block", type=int, default=2, choices=[1, 2, 3])
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--no-mc", action="store_true")
        p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1)))
        p.add_argument("--spy-csv", default=None, help="local SPY daily CSV (date, close) for the buy-and-hold reference line")
        p.add_argument("--bp-strategy", default="iron_condor")
        p.add_argument("--bp-tickers", default="SPY")
        p.add_argument("--bp-profile", default="^green")
        p.add_argument("--bp-dd", type=float, default=0.5)
        p.add_argument("--baseline-tag", default=None)
        p.add_argument("--config-dir", default=None)

    r = sub.add_parser("report", help="build report/index.html and friends")
    r.add_argument("--runs", default="runs")
    r.add_argument("--out", default="report")
    add_report_flags(r)

    d = sub.add_parser("demo", help="make-synthetic-logs + report in one go (SYNTHETIC)")
    d.add_argument("--runs-out", default="runs_synth")
    d.add_argument("--out", default="report_synth")
    d.add_argument("--start", default="2016-10-03")
    d.add_argument("--end", default="2026-09-30")
    add_report_flags(d)

    a = ap.parse_args(argv)
    if a.cmd == "validate":
        from .schema import validate_cli
        sys.exit(validate_cli(a.runs))
    if a.cmd == "make-synthetic-logs":
        from .synth import generate
        generate(a.out, seed=a.seed, start=a.start, end=a.end, cadence=a.cadence, verbose=True)
        print(f"SYNTHETIC logs written to {a.out}/ (fixtures only)")
        return
    from .pipeline import ReportOptions, run_report

    def opts(a):
        return ReportOptions(allocations=a.allocations, apy=a.apy, cb=a.cb, expiry_cap=(a.expiry_cap or None), paths=a.paths, block=a.block,
                             seed=a.seed, no_mc=a.no_mc, jobs=a.jobs, spy_csv=a.spy_csv, bp_strategy=a.bp_strategy,
                             bp_tickers=[t for t in a.bp_tickers.split(",") if t], bp_profile=a.bp_profile, bp_dd=a.bp_dd,
                             baseline_tag=a.baseline_tag, config_dir=a.config_dir)
    if a.cmd == "report":
        run_report(a.runs, a.out, opts(a))
    elif a.cmd == "demo":
        from .synth import generate
        generate(a.runs_out, seed=a.seed, start=a.start, end=a.end)
        print(f"SYNTHETIC logs written to {a.runs_out}/")
        run_report(a.runs_out, a.out, opts(a))
