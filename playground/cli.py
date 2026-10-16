"""python -m playground <command> ..."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from .config import Config, apply_overrides, build_config, profile_names
from .events import DEFAULT_PATH, EventIndex, EventsFileMissing, load_events

S = argparse.SUPPRESS


def onoff(x: str) -> bool:
    if x.lower() not in ("on", "off"):
        raise argparse.ArgumentTypeError("use on|off")
    return x.lower() == "on"


def pct(x: str) -> float:
    v = float(x.rstrip("%"))
    return v / 100 if (x.endswith("%") or v > 1) else v


def add_rule_flags(p: argparse.ArgumentParser):
    g = p.add_argument_group("strategy / rule overrides (defaults come from the profile; Green = proven rules)")
    g.add_argument("--profile", default="green", help=f"one of: {', '.join(profile_names())}")
    g.add_argument("--budget-pct", type=pct, default=S, help="risk budget as %% of equity (green 15, yellow 5, red 1)")
    g.add_argument("--start", dest="start_equity", type=float, default=S, help="starting equity (default 100000)")
    g.add_argument("--apy", type=float, default=S)
    g.add_argument("--full-use", dest="full_use", action="store_true", default=S, help="round contracts to nearest")
    g.add_argument("--min-credit", type=float, default=S)
    g.add_argument("--expiry-cap", dest="expiry_cap", type=onoff, default=S)
    g.add_argument("--expiry-cap-frac", type=float, default=S)
    g.add_argument("--commission", type=float, default=S, help="$ per contract per leg (default 0)")
    g.add_argument("--gate", type=onoff, default=S)
    g.add_argument("--upside-guard", dest="upside_guard", type=onoff, default=S)
    g.add_argument("--upside-pct", dest="upside_5d", type=pct, default=S)
    g.add_argument("--event-buffer", dest="event_buffer", type=onoff, default=S)
    g.add_argument("--cb", dest="cb_days", type=int, choices=[0, 21], default=S)
    g.add_argument("--tp", type=float, default=S)
    g.add_argument("--stop-mult", dest="stop_mult", default=S, help="e.g. 2.0, or 'off'")
    g.add_argument("--breach", dest="breach_exit", type=onoff, default=S)
    g.add_argument("--time-exit", dest="time_exit", type=int, default=S, help="exit at N DTE (0=off)")
    g.add_argument("--dte", default=S, help="min,target,max e.g. 30,35,45")
    g.add_argument("--entry-every", dest="entry_every", type=int, default=S)
    g.add_argument("--strike-mode", choices=["vix", "vix-distance", "delta"], default=S)
    g.add_argument("--short-delta", type=float, default=S)
    g.add_argument("--delta", type=float, default=S, help="short-leg delta for BOTH sides (implies --strike-mode delta)")
    g.add_argument("--put-delta", type=float, default=S)
    g.add_argument("--call-delta", type=float, default=S)
    g.add_argument("--fill", choices=["natural", "mid", "haircut"], default=S)
    g.add_argument("--haircut", type=float, default=S)
    g.add_argument("--entry-window", default=S, help="HH:MM-HH:MM (default 09:45-15:30)")
    g.add_argument("--entry-sample", choices=["first", "random", "every-30m"], default=S)
    g.add_argument("--seed", type=int, default=S)
    g.add_argument("--resolution", choices=["1m", "5m", "15m", "eod"], default=S)
    g.add_argument("--start-date", default=S)
    g.add_argument("--end-date", default=S)
    g.add_argument("--red-ignore", default=S, help="red: comma list of event,expiry-cap,time-exit,stop")


def add_data_flags(p: argparse.ArgumentParser):
    g = p.add_argument_group("data")
    g.add_argument("--provider", choices=["cache", "synthetic"], default="cache")
    g.add_argument("--data-dir", default="data")
    g.add_argument("--results-dir", default="results")
    g.add_argument("--events", default=str(DEFAULT_PATH))
    g.add_argument("--syn-start", default="2015-01-02")
    g.add_argument("--syn-end", default="2025-12-31")
    g.add_argument("--syn-seed", type=int, default=7)


def overrides_from(args, include_profile=False) -> dict:
    keys = {"budget_pct", "start_equity", "apy", "full_use", "min_credit", "expiry_cap", "expiry_cap_frac",
            "commission", "gate", "upside_guard", "upside_5d", "event_buffer", "cb_days", "tp", "stop_mult",
            "breach_exit", "time_exit", "entry_every", "fill", "haircut", "entry_window", "entry_sample",
            "seed", "resolution", "start_date", "end_date",
            # lab engine
            "ticker", "strategy", "sizing", "entry_cadence", "dte_target", "width_mode", "width_pct", "min_credit_pct",
            "fee_per_contract", "spot_source", "spot_fallback", "ifly_tp", "stop_dollar", "forced_close_dte", "trigger",
            "trigger_value", "action", "max_rolls", "roll_credit_only", "roll_to_pct", "hedge_delta", "hedge_expiry",
            "stock_slip_bps", "term_filter", "post_spike_wait", "spike_pct", "min_ivr", "delta_cap_pct", "scale_in",
            "close_before_exdiv", "bwb_near_pct", "bwb_far_pct", "force_entry"}
    ov = {k: v for k, v in vars(args).items() if k in keys}
    if "stop_mult" in ov:
        ov["stop_mult"] = None if str(ov["stop_mult"]).lower() in ("off", "none", "0") else float(ov["stop_mult"])
    a = vars(args)
    if "dte" in a:
        ov["dte"] = [int(x) for x in a["dte"].split(",")]
        if len(ov["dte"]) != 3:
            raise SystemExit("--dte needs min,target,max")
    sr = {}
    if "strike_mode" in a:
        sr["mode"] = "vix" if a["strike_mode"] in ("vix", "vix-distance") else a["strike_mode"]
    if "short_delta" in a:
        sr["put_delta"] = sr["call_delta"] = a["short_delta"]
    if "delta" in a:
        sr["mode"], sr["put_delta"], sr["call_delta"] = "delta", a["delta"], a["delta"]
    if "put_delta" in a:
        sr["mode"], sr["put_delta"] = "delta", a["put_delta"]
    if "call_delta" in a:
        sr["mode"], sr["call_delta"] = "delta", a["call_delta"]
    if sr:
        ov["strike_rule"] = sr
    for item in (a.get("red_ignore") or "").split(","):
        item = item.strip()
        if item:
            ov.update({"event": {"event_buffer": False}, "expiry-cap": {"expiry_cap": False},
                       "time-exit": {"time_exit": 0}, "stop": {"stop_mult": None}}[item])
    return ov


def make_cfg(args, profile: str | None = None) -> Config:
    return build_config(profile or getattr(args, "profile", "green"), overrides_from(args))


def make_provider(args, cfg: Config, fetcher=None):
    if args.provider == "synthetic":
        from .provider import SyntheticProvider
        return SyntheticProvider(start=args.syn_start, end=args.syn_end, seed=args.syn_seed)
    from .cache import DataCache
    from .theta_provider import ThetaProvider
    return ThetaProvider(DataCache(args.data_dir), fetcher)


def get_events(args, cfg: Config, provider, required: bool = True):
    """required=False (used by `plan`): a missing events file only warns; the earnings/FOMC filters are treated as off."""
    need = cfg.event_buffer or cfg.earnings_filter
    if not need:
        return None, "off"
    p = Path(args.events)
    if provider.name == "synthetic":              # never mix a real data/events.csv with the synthetic world's prices
        return provider.synthetic_events(), "SYNTHETIC-calendar(not real dates)"
    if p.exists():
        return load_events(p), str(p)
    if not required:
        print(f"WARNING: events file not found: {p}. Continuing with the earnings/FOMC event filters DISABLED for this "
              "plan (candidate-day counts are an upper bound). Create it from templates/events.csv.template with real "
              "dates to include them.", file=sys.stderr)
        return None, "off (events file missing)"
    raise EventsFileMissing(
        f"events file not found: {p}\nCreate it from templates/events.csv.template with real dates "
        "(see README) or run with --event-buffer off. Dates are never invented.")


# ====================================================================== commands
def cmd_backtest(args):
    from .engine import run_backtest
    from .report import compact_summary, save_run
    cfg = make_cfg(args)
    prov = make_provider(args, cfg)
    ev, note = get_events(args, cfg, prov)
    res = run_backtest(prov, cfg, ev, note)
    out = save_run(res, args.results_dir, getattr(args, "run_id", None))
    print(compact_summary(res))
    print(f"saved: {out}")


def cmd_matrix(args):
    from .ladder import run_matrix
    cfg = make_cfg(args)
    prov = make_provider(args, cfg)
    ev, note = get_events(args, cfg, prov)
    print(run_matrix(prov, cfg, ev, note, fill_compare=args.fill_compare))


def cmd_ladder(args):
    from .ladder import run_ladder
    cfg = make_cfg(args, "green")
    prov = make_provider(args, cfg)
    ev, note = get_events(args, cfg, prov)
    ov = overrides_from(args)
    print(run_ladder(prov, ov, ev, note, equal_budget=args.equal_budget, split_sample=args.split_sample))


def _client(args):
    from .theta_client import ThetaClient
    if getattr(args, "fake", False):
        from .fake_theta import FakeSession
        from .provider import SyntheticProvider
        return ThetaClient("http://fake", max_rps=0,
                           session=FakeSession(SyntheticProvider(start=args.syn_start, end=args.syn_end)))
    return ThetaClient(max_rps=getattr(args, "max_rps", 5.0))


def cmd_probe(args):
    from .probe import Probe
    c = _client(args)
    rep = Probe(c, out_dir=args.results_dir, data_dir=args.data_dir,
                tickers=[t.strip().upper() for t in args.tickers.split(",") if t.strip()]).run()
    bad = [x for x in rep["checks"] if x["level"] == "FAIL"]
    info = sum(x["level"] == "INFO" for x in rep["checks"])
    print(f"probe: {len(rep['checks']) - len(bad) - info} passed, {len(bad)} failed, {info} info -> {args.results_dir}/probe_report.md")
    for x in bad:
        print(f"FAIL [{x['section']}] {x['name']}: {x['detail']} | {x['endpoint']} {x['params']} "
              f"status={x['status']} raw={x['raw_error'][:300]!r}")
    if not rep["checks"]:
        print("no checks ran")


def cmd_plan(args):
    if getattr(args, "engine", "green") == "lab":
        from .cli_lab import cmd_plan_lab
        return cmd_plan_lab(args)
    from .planner import make_plan
    cfg = make_cfg(args)
    prov = make_provider(args, cfg)
    ev, note = get_events(args, cfg, prov, required=False)
    daily, vix = prov.get_underlying_daily(), prov.get_vix_daily()
    evx = EventIndex(ev, list(daily.index)) if ev is not None else None
    print(make_plan(cfg, daily, vix, evx, threads=(2, 4), max_rps=args.max_rps, entry_fetch=args.entry_fetch,
                    chunk_days=args.chunk_days, concurrency_limit=args.concurrency_limit, results_dir=args.results_dir, data_dir=args.data_dir,
                    attempt_frac=args.attempt_frac, trades_per_year=args.trades_per_year))


def cmd_download(args):
    if getattr(args, "engine", "green") == "lab":
        from .cli_lab import cmd_download_lab
        return cmd_download_lab(args)
    from .cache import DataCache
    from .dailydata import ensure_daily
    from .downloader import Fetcher, stage_a, write_manifest
    from .engine import run_backtest
    from .ladder import matrix_configs
    from .theta_provider import ThetaProvider
    cfg = make_cfg(args)
    cache = DataCache(args.data_dir)
    from .downloader import make_logger
    log = make_logger(cache.root / "state" / "download.log")

    client = _client(args)
    end = date.fromisoformat(cfg.end_date) if cfg.end_date else date.today() - timedelta(days=1)
    start = date.fromisoformat(cfg.start_date) if cfg.start_date else end - timedelta(days=365 * args.years)
    start_data = start - timedelta(days=30)
    if args.fake:
        from .provider import SyntheticProvider
        syn = SyntheticProvider(start=args.syn_start, end=args.syn_end)
        cache.save_daily("SPY", syn.get_underlying_daily())
        cache.save_daily("VIX", syn.get_vix_daily().to_frame("close"))
        info = {"spy_source": "fake", "vix_source": "fake"}
    else:
        info = ensure_daily(cache, client, start_data, end, log)
        from .dailydata import ensure_vol_indices
        info["vol_indices"] = ensure_vol_indices(cache, log)
    log("daily data:", info)
    tmp = ThetaProvider(cache)
    try:
        daily = tmp.get_underlying_daily()
    except Exception as e:
        print(f"ERROR: {e}")
        return 2
    if args.daily_only:
        return 0
    if args.threads > args.concurrency_limit:
        log(f"--threads {args.threads} exceeds the {args.concurrency_limit}-concurrent-request limit; clamping")
    fetcher = Fetcher(client, cache, list(daily.index), threads=min(args.threads, args.concurrency_limit), chunk_days=args.chunk_days,
                      entry_fetch=args.entry_fetch, log=log)
    prov = ThetaProvider(cache, fetcher)
    cfg = apply_overrides(cfg, {"start_date": cfg.start_date or str(start), "end_date": cfg.end_date or str(end)})
    ev, note = (None, "off")
    if cfg.event_buffer or cfg.earnings_filter:
        p = Path(args.events)
        if not p.exists() and not args.fake:
            print(f"ERROR: events file not found: {p}. Create it from templates/events.csv.template "
                  "(real dates) or pass --event-buffer off.")
            return 2
        ev, note = (load_events(p), str(p)) if (p.exists() and not args.fake) else (SyntheticProvider(
            start=args.syn_start, end=args.syn_end).synthetic_events(), "SYNTHETIC")
    evx = EventIndex(ev, list(daily.index)) if ev is not None else None
    status = {}
    try:
        if not fetcher.cache.expirations():
            fetcher.fetch_expirations()
        cover = cfg
        if args.cover_matrix:   # superset of every matrix variant's entry days
            cover = apply_overrides(cfg, {"gate": False, "upside_guard": False, "cb_days": 0})
        if not args.no_stage_a:
            status["stage_a"] = stage_a(prov, fetcher, cover, evx, log, threads=args.threads, limit=args.limit_days,
                                       progress_every=args.progress_every)
        cfgs = [("run", cfg)] if not args.cover_matrix else list(matrix_configs(cfg))
        for name, c in cfgs:
            log(f"stage B / on-demand pass: {name}")
            r = run_backtest(prov, c, ev, note)
            status.setdefault("passes", []).append((name, len(r.trades), r.gaps))
    except KeyboardInterrupt:
        log("interrupted - cache is consistent; re-run to resume")
    m = write_manifest(cache, fetcher, {"status": status})
    errs = status.get("stage_a", {}).get("errors", [])
    print("MANIFEST", {k: (v if k != "failed" else f"{len(v)} failed") for k, v in m.items() if k != "status"})
    print("stage A:", {k: (v if k != "errors" else f"{len(v)} days with errors") for k, v in status.get("stage_a", {}).items()})
    print("missing/failed days:", [str(d) for d, _ in errs][:30], "(first errors: %s)" % [e for _, e in errs[:2]])
    print("passes (name, trades, data gaps):", status.get("passes"))
    return 0


def cmd_stub(args):
    from . import reserve
    getattr(reserve, args.cmd.replace("-", "_"))()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m playground", description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("probe", help="check what the local Theta Terminal provides (writes results/probe_report.md)")
    add_data_flags(p)
    p.add_argument("--fake", action="store_true", help="use the offline simulator (tests the plumbing only)")
    p.add_argument("--tickers", default="SPY,QQQ,IWM,GLD,TLT")
    p.set_defaults(fn=cmd_probe)

    from .cli_lab import add_lab_flags
    p = sp.add_parser("plan", help="dry-run estimate of download size/time (writes nothing to data/)")
    add_rule_flags(p); add_data_flags(p); add_lab_flags(p)
    p.add_argument("--max-rps", type=float, default=5.0)
    p.add_argument("--entry-fetch", choices=["per-strike", "bulk"], default="per-strike")
    p.add_argument("--chunk-days", type=int, default=10)
    p.add_argument("--concurrency-limit", type=int, default=4, help="Standard 4, Value 2")
    p.add_argument("--attempt-frac", type=float, default=0.35)
    p.add_argument("--trades-per-year", type=float, default=30)
    p.add_argument("--engine", choices=["green", "lab"], default="green", help="lab: per-ticker/per-cadence plan")
    p.add_argument("--tickers", default="SPY,QQQ,IWM,GLD,TLT")
    p.add_argument("--entry-cadences", default="monthly,weekly,twice-weekly")
    p.set_defaults(fn=cmd_plan)

    p = sp.add_parser("download", help="resumable two-stage downloader into data/")
    add_rule_flags(p); add_data_flags(p); add_lab_flags(p)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--concurrency-limit", type=int, default=4, help="Standard 4, Value 2")
    p.add_argument("--max-rps", type=float, default=5.0)
    p.add_argument("--entry-fetch", choices=["per-strike", "bulk"], default="per-strike")
    p.add_argument("--chunk-days", type=int, default=10)
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--daily-only", action="store_true")
    p.add_argument("--no-stage-a", action="store_true")
    p.add_argument("--cover-matrix", action="store_true", help="fetch what all 8 matrix variants need")
    p.add_argument("--limit-days", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=50, help="print a progress line (entry date, count, total, ETA) every N requests")
    p.add_argument("--fake", action="store_true", help="offline simulator instead of a Theta Terminal")
    p.add_argument("--engine", choices=["green", "lab"], default="green")
    p.add_argument("--tickers", default="SPY")
    p.add_argument("--max-gb", type=float, default=None, help="hard stop per ticker (GB on disk)")
    p.add_argument("--max-hours", type=float, default=None, help="hard stop per ticker (wall-clock hours)")
    p.set_defaults(fn=cmd_download)

    p = sp.add_parser("backtest", help="run one backtest; compact summary + results/<run_id>/")
    add_rule_flags(p); add_data_flags(p)
    p.add_argument("--run-id", default=S)
    p.set_defaults(fn=cmd_backtest)

    p = sp.add_parser("matrix", help="{gate off, gate, gate+upside, +cb21} x {exit A, exit B}")
    add_rule_flags(p); add_data_flags(p)
    p.add_argument("--fill-compare", action="store_true", help="show natural and mid side by side")
    p.set_defaults(fn=cmd_matrix)

    p = sp.add_parser("ladder", help="traffic-light ladder: green, yellow_E1..E7, red")
    add_rule_flags(p); add_data_flags(p)
    p.add_argument("--equal-budget", action="store_true", help="give every profile Green's budget")
    p.add_argument("--split-sample", action="store_true", help="first half vs second half")
    p.set_defaults(fn=cmd_ladder)

    from .cli_lab import register
    register(sp)
    for name in ("overlay", "spike-study"):
        p = sp.add_parser(name, help="RESERVED - not built")
        p.set_defaults(fn=cmd_stub)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args) or 0
    except EventsFileMissing as e:
        print(f"ERROR: {e}")
        return 2
    except NotImplementedError as e:
        print(f"NOT BUILT: {e}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
