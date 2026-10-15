"""CLI commands for the lab engine: run, sweep, stress, spot, export-outputs, purge-vendor-data (+ lab download/plan)."""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path


from . import factory
from .config import apply_overrides
from .events import EventIndex, load_events
from .report import table

S = argparse.SUPPRESS


def onoff(x: str) -> bool:
    if x.lower() not in ("on", "off"):
        raise argparse.ArgumentTypeError("use on|off")
    return x.lower() == "on"


def pctv(x: str) -> float:
    v = float(str(x).rstrip("%"))
    return v / 100 if (str(x).endswith("%") or v > 1) else v


def add_lab_flags(p: argparse.ArgumentParser):
    g = p.add_argument_group("lab engine (multi-ticker / multi-strategy / management)")
    g.add_argument("--ticker", default=S, help="SPY QQQ IWM GLD TLT")
    g.add_argument("--strategy", default=S, help="iron_condor iron_butterfly bull_put_spread put_bwb cash_secured_put covered_call")
    g.add_argument("--sizing", choices=["equity-pct", "fixed1"], default=S, help="fixed1 = unit mode (1 contract, logs for the replay layer)")
    g.add_argument("--entry-cadence", dest="entry_cadence", choices=["monthly", "weekly", "twice-weekly", "daily"], default=S)
    g.add_argument("--dte-target", dest="dte_target", type=int, choices=[30, 45, 60, 90], default=S)
    g.add_argument("--width-mode", dest="width_mode", choices=["auto", "pct", "fixed"], default=S)
    g.add_argument("--width-pct", dest="width_pct", type=pctv, default=S, help="spread width as %% of spot (default 0.7)")
    g.add_argument("--min-credit-pct", dest="min_credit_pct", type=pctv, default=S, help="min credit as %% of width (default 10)")
    g.add_argument("--fee-per-contract", dest="fee_per_contract", type=float, default=S, help="USD per contract per leg (default 0.05)")
    g.add_argument("--spot-source", dest="spot_source", choices=["alpaca", "parity", "daily"], default=S)
    g.add_argument("--spot-fallback", dest="spot_fallback", choices=["parity", "none"], default=S)
    g.add_argument("--ifly-tp", dest="ifly_tp", type=pctv, default=S, help="iron butterfly take-profit: 25|50")
    g.add_argument("--stop-dollar", dest="stop_dollar", type=float, default=S)
    g.add_argument("--forced-close-dte", dest="forced_close_dte", type=int, default=S)
    g.add_argument("--trigger", choices=["distance", "delta"], default=S)
    g.add_argument("--trigger-value", dest="trigger_value", type=float, default=S)
    g.add_argument("--action", default=S, help="none exit roll_tested roll_condor roll_untested close_tested_spread hedge_put_call hedge_shares")
    g.add_argument("--max-rolls", dest="max_rolls", type=int, default=S)
    g.add_argument("--roll-credit-only", dest="roll_credit_only", type=onoff, default=S)
    g.add_argument("--roll-to-pct", dest="roll_to_pct", type=pctv, default=S)
    g.add_argument("--hedge-delta", dest="hedge_delta", type=float, default=S)
    g.add_argument("--hedge-expiry", dest="hedge_expiry", choices=["same", "next"], default=S)
    g.add_argument("--stock-slip-bps", dest="stock_slip_bps", type=float, default=S)
    g.add_argument("--term-filter", dest="term_filter", type=onoff, default=S, help="block when VIX/VIX3M > 1.0")
    g.add_argument("--post-spike-wait", dest="post_spike_wait", type=int, default=S)
    g.add_argument("--spike-pct", dest="spike_pct", type=pctv, default=S)
    g.add_argument("--min-ivr", dest="min_ivr", type=float, choices=[0, 30, 50], default=S)
    g.add_argument("--delta-cap", dest="delta_cap_pct", type=float, default=S, help="delta-dollars <= X%% of capital at risk (150|200)")
    g.add_argument("--scale-in", dest="scale_in", type=int, choices=[1, 4], default=S)
    g.add_argument("--close-before-exdiv", dest="close_before_exdiv", type=onoff, default=S)
    g.add_argument("--bwb-near-pct", dest="bwb_near_pct", type=pctv, default=S)
    g.add_argument("--bwb-far-pct", dest="bwb_far_pct", type=pctv, default=S)
    g.add_argument("--force-entry", dest="force_entry", action="store_true", default=S)
    g.add_argument("--runs-dir", default="runs")


def _cmdline() -> str:
    return "python -m playground " + " ".join(sys.argv[1:])


def _prov(args, cfg):
    return factory.make(args.provider, cfg.ticker, args.data_dir, args.syn_start, args.syn_end, args.syn_seed, rate=cfg.apy)


def _events(args, cfg, prov):
    from .cli import get_events
    return get_events(args, cfg, prov)


def cmd_run(args):
    from .cli import make_cfg
    from .lab import run_lab, save_lab_run
    from .schema import validate_run_dir
    cfg = make_cfg(args)
    if not hasattr(args, "sizing"):
        cfg = cfg.replace(sizing="fixed1")
    prov = _prov(args, cfg)
    ev, note = _events(args, cfg, prov)
    res = run_lab(prov, cfg, ev, note)
    out = save_lab_run(res, args.runs_dir, _cmdline())
    t, m = res.trades, res.meta
    L = [f"{cfg.profile.upper()} (level {cfg.level}) | {cfg.ticker} {cfg.strategy} | tag={res.config_tag} | data={prov.name} | "
         f"fill={cfg.fill} fee={cfg.fee_per_contract}/leg | sizing={cfg.sizing} | cadence={cfg.entry_cadence} | events={note}"]
    if len(t):
        L.append(f"trades {len(t)} | win {float((t.pnl_per_contract > 0).mean()):.0%} | avg pnl {t.pnl_per_contract.mean():.1f}$/c | "
                 f"total {t.pnl_per_contract.sum():.0f}$/c | avg credit {t.credit_per_contract.mean():.1f}$ | avg collateral {t.collateral_per_contract.mean():.0f}$")
    else:
        L.append("trades 0")
    L.append("exits: " + " ".join(f"{k}={v}" for k, v in m["counts"]["exit_reasons"].items()))
    L.append("blocked: " + " ".join(f"{k}={v}" for k, v in m["counts"]["blocked"].items()))
    L.append(f"spot: {m['spot_coverage']['days_by_source']} no-spot days={m['spot_coverage']['days_without_spot']} | "
             f"delta source: {m['delta_source']}")
    errs = validate_run_dir(out)
    L.append(f"saved {out} | schema {'OK' if not errs else 'PROBLEMS: ' + '; '.join(errs[:3])}")
    print("\n".join(L))
    return 0 if not errs else 1


def cmd_sweep(args):
    from .sweep import format_summary, load_spec, run_sweep
    spec = load_spec(args.spec)
    res = run_sweep(spec, args, args.runs_dir, command=_cmdline())
    print(format_summary(res))
    return 0


def cmd_stress(args):
    from .cli import overrides_from
    from .stress import load_windows, run_stress
    ws = load_windows(args.windows)
    if args.only:
        ws = [w for w in ws if w["name"] in args.only.split(",")]
    ov = overrides_from(args)
    rows = run_stress(args, ws, getattr(args, "profile", "green"), ov, every_day=args.every_day,
                      runs_dir=args.runs_dir, command=_cmdline(), log=lambda *a: None)
    print("STRESS (windows are APPROXIMATE month-level ranges from data/stress_windows.csv; forced entries, unit mode)")
    print(table([[r["window"], r["trades"], f"{r['pnl_total']:.0f}", f"{r['worst']:.0f}", r["run_id"]] for r in rows],
                ["window", "trades", "total $/contract", "worst $/contract", "run_id"]))
    return 0


def cmd_export(args):
    from .housekeeping import export_outputs
    p = export_outputs(".", args.out_dir)
    print(f"wrote {p} (no data/, cache/ or parquet files included)")
    return 0


def cmd_purge(args):
    from .housekeeping import purge_vendor_data
    res = purge_vendor_data(args.data_dir, yes=args.yes, include_spot=args.include_spot, log_path=args.log)
    print(table([[p, n, f"{b / 1e6:.1f} MB"] for p, n, b in res["targets"]] or [["(nothing to delete)", 0, "0"]],
                ["vendor path", "files", "size"]))
    if res["dry_run"]:
        print("DRY RUN - nothing deleted. Re-run with --yes to delete (and write purge_log.md).")
    else:
        print(f"deleted {len(res['deleted'])} directories; certificate: {res['log']}")
    return 0


def cmd_spot(args):
    from . import spot as sp
    from .dailydata import ensure_vol_indices
    from .cache import DataCache
    from .exdiv import fetch_exdiv, save_exdiv
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    end = date.fromisoformat(args.end_date) if args.end_date else date.today() - timedelta(days=1)
    start = date.fromisoformat(args.start_date) if args.start_date else end - timedelta(days=365 * 10)
    store = sp.SpotStore(Path(args.data_dir) / "spot")
    try:
        client = sp.AlpacaClient()
        if not client.key or not client.secret:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set (use .env, never the repo)")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 2
    if not args.coverage_only:
        print("downloading Alpaca 1m bars (UNVERIFIED path) ...", flush=True)
        summ = sp.download_spot(client, store, tickers, start, end, log=lambda m: print(m, file=sys.stderr))
        print("downloaded:", summ)
        for t in tickers:
            ds, src = fetch_exdiv(t, start, end, client)
            if ds:
                save_exdiv(store.root, t, ds)
            print(f"ex-dividend dates {t}: {len(ds)} from {src}")
        ensure_vol_indices(DataCache(args.data_dir), print)
    covs = {t: sp.coverage(store, t, start, end) for t in tickers}
    sp.write_coverage_md(covs, Path(args.results_dir) / "spot_coverage.md")
    for t, df in covs.items():
        print(f"coverage {t}: ok={int((df.status == 'ok').sum())} flagged={int((df.status != 'ok').sum())} of {len(df)} days")
    print(f"wrote {args.results_dir}/spot_coverage.md")
    if args.validate:
        rows, notes = sp.run_validations(args, store, tickers, start, end, client)
        sp.write_validation_md(rows, Path(args.results_dir) / "spot_validation.md", notes)
        print(f"wrote {args.results_dir}/spot_validation.md")
    return 0


def cmd_download_lab(args):
    from .cache import DataCache
    from .cli import _client, make_cfg
    from .dailydata import ensure_daily, ensure_vol_indices
    from .downloader import BudgetExceeded, Fetcher, stage_a, write_manifest
    from .lab import run_lab
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    base = make_cfg(args)
    client = _client(args)
    root = DataCache(args.data_dir)
    log = lambda *a: print(*a, file=sys.stderr, flush=True)
    end = date.fromisoformat(base.end_date) if base.end_date else date.today() - timedelta(days=1)
    start = date.fromisoformat(base.start_date) if base.start_date else end - timedelta(days=365 * args.years)
    if not args.fake:
        ensure_daily(root, client, start - timedelta(days=30), end, log)
        ensure_vol_indices(root, log)
    else:
        syn = factory.synthetic("SPY", args.syn_start, args.syn_end, args.syn_seed)
        root.save_daily("SPY", syn.get_underlying_daily())
        root.save_daily("VIX", syn.get_vix_daily().to_frame("close"))
        root.save_daily("VIX3M", syn.get_vix3m_daily().to_frame("close"))
    ev = evx = None
    note = "off"
    if base.event_buffer or base.earnings_filter:
        p = Path(args.events)
        if p.exists():
            ev, note = load_events(p), str(p)
        elif args.fake:
            ev, note = factory.synthetic("SPY", args.syn_start, args.syn_end, args.syn_seed).synthetic_events(), "SYNTHETIC"
        else:
            print(f"ERROR: events file not found: {p}. Create it from templates/events.csv.template (real dates) or pass --event-buffer off.")
            return 2
    for t in tickers:
        cfg = apply_overrides(base, {"ticker": t, "start_date": str(start), "end_date": str(end)})
        cache = factory.ticker_cache(args.data_dir, t)
        if args.fake:
            syn = factory.synthetic(t, args.syn_start, args.syn_end, args.syn_seed)
            cache.save_daily(t, syn.get_underlying_daily())
            client.s.provs[t] = syn if hasattr(client.s, "provs") else None
        prov0 = factory.theta(t, args.data_dir, None, cfg.apy)
        try:
            daily = prov0.get_underlying_daily()
        except Exception as e:
            print(f"ERROR {t}: {e} (run `spot` or `download --daily-only` first)")
            continue
        fetcher = Fetcher(client, cache, list(daily.index), threads=min(args.threads, args.concurrency_limit),
                          chunk_days=args.chunk_days, entry_fetch="bulk" if cfg.strike_rule["mode"] == "delta" else args.entry_fetch,
                          log=log, symbol=t)
        fetcher.set_limits(args.max_gb, args.max_hours)
        prov = factory.theta(t, args.data_dir, fetcher, cfg.apy)
        evx = EventIndex(ev, list(daily.index), t) if ev is not None else None
        status = {}
        try:
            if not cache.expirations():
                fetcher.fetch_expirations()
            status["stage_a"] = stage_a(prov, fetcher, cfg, evx, log, threads=args.threads, limit=args.limit_days, lab=True)
            if not status["stage_a"]["stopped"]:
                res = run_lab(prov, apply_overrides(cfg, {"sizing": "fixed1"}), ev, note)
                status["stage_b_pass"] = {"trades": len(res.trades), "gaps": res.meta["spot_coverage"]["days_without_spot"]}
        except BudgetExceeded as e:
            status["stopped"] = str(e)
        except KeyboardInterrupt:
            log("interrupted - cache is consistent; re-run to resume")
        m = write_manifest(cache, fetcher, {"ticker": t, "status": status})
        print(f"MANIFEST {t}:", {k: v for k, v in m.items() if k not in ("failed", "status")})
        print(f"  status: { {k: (v if k != 'stage_a' else {a: b for a, b in v.items() if a != 'errors'}) for k, v in status.items()} }")
    return 0


def cmd_plan_lab(args):
    from .cli import get_events, make_cfg
    from .planner import make_lab_plan
    base = make_cfg(args)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    dailies, ev, vix, note = {}, None, None, "off"
    for t in tickers:
        prov = factory.make(args.provider, t, args.data_dir, args.syn_start, args.syn_end, args.syn_seed, rate=base.apy)
        try:
            dailies[t] = prov.get_underlying_daily()
        except Exception as e:
            print(f"{t}: skipped ({e})")
            continue
        if vix is None:
            vix = prov.get_vix_daily()
            ev, note = get_events(args, base, prov)
    if not dailies:
        print("no daily data - run `spot` / `download --daily-only` first")
        return 2
    first = next(iter(dailies.values()))
    evx = EventIndex(ev, list(first.index)) if ev is not None else None
    print(make_lab_plan(base, dailies, vix, evx, [c.strip() for c in args.entry_cadences.split(",")],
                        max_rps=args.max_rps, concurrency_limit=args.concurrency_limit, chunk_days=args.chunk_days,
                        results_dir=args.results_dir, data_dir=args.data_dir))
    return 0


def register(sp):
    from .cli import add_data_flags, add_rule_flags
    p = sp.add_parser("run", help="one lab-engine run (default unit mode) -> runs/<run_id>/")
    add_rule_flags(p); add_data_flags(p); add_lab_flags(p)
    p.set_defaults(fn=cmd_run)

    p = sp.add_parser("sweep", help="resumable parameter grid from a YAML spec (unit mode) -> runs/")
    add_data_flags(p)
    p.add_argument("spec")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--limit", type=int, default=None, help="run only the first N configurations (quick checks)")
    p.set_defaults(fn=cmd_sweep)

    p = sp.add_parser("stress", help="forced-entry replay through data/stress_windows.csv (unit mode)")
    add_rule_flags(p); add_data_flags(p); add_lab_flags(p)
    p.add_argument("--windows", default="data/stress_windows.csv")
    p.add_argument("--only", default="")
    p.add_argument("--every-day", action="store_true")
    p.set_defaults(fn=cmd_stress)

    p = sp.add_parser("spot", help="download/validate intraday spot (Alpaca primary) - UNVERIFIED until run on the VPS")
    p.add_argument("--tickers", default="SPY,QQQ,IWM,GLD,TLT")
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--coverage-only", action="store_true")
    p.add_argument("--validate", default="", help="comma list: massive,greeks,parity")
    p.add_argument("--sample-days", type=int, default=6)
    p.add_argument("--minutes-per-day", type=int, default=25)
    p.add_argument("--mark-bad", type=onoff, default=True, help="flag days with median abs diff > $0.25 so the engine falls back")
    p.set_defaults(fn=cmd_spot)

    p = sp.add_parser("export-outputs", help="tar.gz of derived outputs (never data/, cache/ or parquet)")
    p.add_argument("--out-dir", default="exports")
    p.set_defaults(fn=cmd_export)

    p = sp.add_parser("purge-vendor-data", help="delete ThetaData caches (dry run unless --yes); writes purge_log.md")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--include-spot", action="store_true", help="also delete Alpaca/Massive spot caches")
    p.add_argument("--log", default="purge_log.md")
    p.set_defaults(fn=cmd_purge)
