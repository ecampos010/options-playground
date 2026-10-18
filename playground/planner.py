"""`plan`: dry-run estimate of the tight two-stage download. Writes nothing to data/."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .config import Config
from .downloader import candidate_days
from .report import table

RES = ["1m", "5m", "15m", "eod"]


def bars_per_day(res: str) -> int:
    return 1 if res == "eod" else 390 // int(res.rstrip("m")) + 1


def load_latency(results_dir="results") -> tuple[float, str]:
    p = Path(results_dir) / "probe_report.json"
    if p.exists():
        try:
            lat = json.loads(p.read_text()).get("measured", {}).get("option_quote_latency_p50_s")
            if lat:
                return float(lat), f"measured by last probe ({p})"
        except Exception:
            pass
    return 0.5, "ASSUMED 0.5s (no probe_report.json yet - run `probe` on the VPS)"


def bytes_per_row(data_dir="data") -> tuple[float, str]:
    p = Path(data_dir) / "state" / "manifest.json"
    if p.exists():
        try:
            m = json.loads(p.read_text())
            if m.get("rows_written", 0) > 200_000 and m.get("cache_bytes_on_disk"):
                return m["cache_bytes_on_disk"] / m["rows_written"], "calibrated from last download manifest"
        except Exception:
            pass
    return 14.0, "ASSUMED 14 B/row (zstd parquet, bid/ask/sizes/ts)"


def make_plan(cfg: Config, daily, vix, evx, *, threads=(2, 4), max_rps=5.0, concurrency_limit=4, value_limit=2, attempt_frac=0.35,
              trades_per_year=30, avg_hold_cal_days=15, chunk_days=10, entry_fetch="per-strike",
              results_dir="results", data_dir="data") -> str:
    cands = candidate_days(cfg, daily, vix, evx)
    all_days = [d for d in daily.index if (not cfg.start_date or str(d) >= cfg.start_date)
                and (not cfg.end_date or str(d) <= cfg.end_date)]
    years = max(len(all_days) / 252.0, 1e-9)
    attempts = len(cands) * attempt_frac
    trades = years * trades_per_year
    lat, lat_src = load_latency(results_dir)
    bpr, bpr_src = bytes_per_row(data_dir)
    n_exp = cfg.entry_expiries
    strikes_per_side = 2 * cfg.band_strikes + 1 + int(cfg.width)
    n_entry_bars = len(__import__("playground.engine", fromlist=["x"]).entry_minutes(all_days[0], cfg)) if all_days else 1
    n_trial = n_exp * 1.5 + 1      # expiry trial quotes (some candidates answer 472) + one strike-grid probe
    lines = [f"PLAN {cfg.start_date or all_days[0]}..{cfg.end_date or all_days[-1]} | "
             f"{len(all_days)} trading days | {len(cands)} candidate entry days after gate/guards/events "
             f"(upper bound for Stage A) | entry sample={cfg.entry_sample} ({n_entry_bars} bar(s)/day)",
             f"assumptions: {attempt_frac:.0%} of candidate days actually attempt an entry (budget/cb), "
             f"{trades_per_year} trades/yr, hold ~{avg_hold_cal_days} cal days, Stage-B chunk {chunk_days} cal days, "
             f"entry-fetch={entry_fetch}, {n_exp} expiries (+~{n_trial:.0f} trial/probe quotes per attempt day, cached; "
             f"HTTP 472 NO_DATA answers count as requests but are cached), {strikes_per_side} strikes/side/expiry",
             f"concurrency: Standard allows {concurrency_limit} concurrent requests, Value {value_limit}; "
             f"wall@Nthr is clamped to the limit (wall@2thr == Value tier)",
             f"latency: {lat:.2f}s/request - {lat_src}; bytes/row: {bpr:.0f} - {bpr_src}; max_rps={max_rps}",
             f"expected: ~{attempts:.0f} attempt days (Stage A on demand) or {len(cands)} if prefetching all "
             f"candidates (default download); ~{trades:.0f} trades"]
    rows = []
    for res in RES:
        B = bars_per_day(res)
        if entry_fetch == "bulk":
            a_req_day = n_exp + n_exp * 1.0 / 20 + 1 / 20 + n_trial                    # + strike listing + bars chunk
            a_rows_day = n_exp * 1600 * n_entry_bars
        else:
            a_req_day = n_exp * 2 * strikes_per_side + n_exp + 1 / 20 + n_trial
            a_rows_day = n_exp * 2 * strikes_per_side * n_entry_bars
        chunks = math.ceil(avg_hold_cal_days / chunk_days)
        b_req_trade = 4 * chunks
        b_rows_trade = 4 * (avg_hold_cal_days * 5 / 7) * B
        for label, nA in (("prefetch-all", len(cands)), ("on-demand", attempts)):
            req = nA * a_req_day + trades * b_req_trade
            nrows = nA * a_rows_day + trades * b_rows_trade + nA * B
            gb = nrows * bpr / 1e9
            times = []
            for t in threads:
                secs = max(req * lat / min(t, concurrency_limit), req / max_rps)
                times.append(f"{secs / 3600:.2f}h")
            rows.append([res, label, f"{req:,.0f}", f"{nrows / 1e6:,.1f}M", f"{gb:.2f}"] + times)
    # naive: every trading day, all expiries up to 60 DTE (daily expiries ~ 42), full chain ~ 1400 contracts
    naive_exp, naive_contracts = 42, 1400
    nrows = []
    for res in RES:
        B = bars_per_day(res)
        r_rows = len(all_days) * naive_exp * naive_contracts * B
        reqs = len(all_days) * naive_exp  # one bulk request per day/expiry
        nrows.append([res, f"{reqs:,.0f} (bulk) / {reqs * naive_contracts:,.0f} (per-contract)",
                      f"{r_rows / 1e9:,.1f}B", f"{r_rows * bpr / 1e9:,.0f}",
                      f"{reqs * lat / 2 / 3600:.0f}h bulk @2thr"])
    head = ["res", "mode", "requests", "rows", "GB"] + [f"wall@{t}thr" for t in threads]
    if lat_src.startswith("ASSUMED"):
        lines.append("WARNING: latency is an assumption; run `probe` (successful quote requests) for a real number")
    out = lines + [table(rows, head), "",
                   "NAIVE full-chain comparison (all days, ~42 expiries, ~1400 contracts each):",
                   table(nrows, ["res", "requests", "rows", "GB", "wall-clock"]),
                   "month budget: 30 days = 720h. Numbers marked ASSUMED are not measured."]
    return "\n".join(out)


LEGS_BY_STRATEGY = {"iron_condor": 4, "iron_butterfly": 4, "bull_put_spread": 2, "put_bwb": 3,
                    "cash_secured_put": 1, "covered_call": 1}


def make_lab_plan(cfg: Config, daily_by_ticker: dict, vix, evx, cadences, *, max_rps=5.0, concurrency_limit=4,
                  value_limit=2, chunk_days=10, avg_hold_cal_days=21, results_dir="results", data_dir="data") -> str:
    """Per-ticker x per-cadence estimates for the lab engine (Stage A bulk/per-strike + Stage B legs)."""
    from .lab import entry_schedule
    lat, lat_src = load_latency(results_dir)
    bpr, bpr_src = bytes_per_row(data_dir)
    delta_mode = cfg.strike_rule["mode"] == "delta"
    B = bars_per_day(cfg.resolution)
    legs = LEGS_BY_STRATEGY.get(cfg.strategy, 4)
    L = [f"LAB PLAN strategy={cfg.strategy} mode={'delta(bulk snapshot)' if delta_mode else 'vix-distance(per-strike band)'} "
         f"dte_target={cfg.dte_target or cfg.dte[1]} res={cfg.resolution} legs/trade={legs}",
         f"latency {lat:.2f}s/request ({lat_src}); {bpr:.0f} B/row ({bpr_src}); limits: Standard {concurrency_limit} "
         f"concurrent / Value {value_limit}; max_rps={max_rps}",
         "Assumptions (not measured): hold ~%d calendar days; Stage-B chunk %d days; bulk snapshot ~1600 rows/request; "
         "+1 trial quote and +1 Greeks request per expiry (cached)." % (avg_hold_cal_days, chunk_days)]
    rows = []
    for ticker, daily in daily_by_ticker.items():
        for cad in cadences:
            c = cfg.replace(ticker=ticker, entry_cadence=cad, entry_every=1)
            alld = list(daily.index)
            i0 = max(6, next((j for j, x in enumerate(alld) if not c.start_date or str(x) >= c.start_date), 6))
            i1 = max([j for j, x in enumerate(alld) if not c.end_date or str(x) <= c.end_date] or [len(alld) - 1])
            cc = c.replace(dte=[c.dte_target - 7, c.dte_target, c.dte_target + 7]) if c.dte_target else c
            sched, _ = entry_schedule(alld, i0, i1, cc)
            cands = set(candidate_days(c, daily, vix, evx))
            entries = len([d for d in sched if d in cands])
            years = max((alld[i1] - alld[i0]).days / 365.25, 1e-9)
            n_exp = 1 if (cad in ("monthly", "biweekly") or cfg.scale_in == 4) else cfg.entry_expiries
            strikes_side = 2 * cfg.band_strikes + 1 + 5
            a_req = n_exp * ((2 + 1) if delta_mode else (1 + 2 * strikes_side))      # trial+bulk+greeks | trial+bands
            a_rows = n_exp * (1600 if delta_mode else 2 * strikes_side)
            b_req = legs * math.ceil(avg_hold_cal_days / chunk_days)
            b_rows = legs * avg_hold_cal_days * 5 / 7 * B
            req = entries * (a_req + b_req)
            nrows = entries * (a_rows + b_rows)
            t2 = max(req * lat / min(2, concurrency_limit), req / max_rps) / 3600
            t4 = max(req * lat / min(4, concurrency_limit), req / max_rps) / 3600
            rows.append([ticker, cad, entries, f"{entries / years:.0f}/yr", f"{req:,.0f}", f"{nrows / 1e6:,.1f}M",
                         f"{nrows * bpr / 1e9:.2f}", f"{t2:.2f}h", f"{t4:.2f}h"])
    L.append(table(rows, ["ticker", "cadence", "entries", "rate", "requests", "rows", "GB", "wall@2thr(Value)", "wall@4thr(Standard)"]))
    L.append("Entry counts come from the cached daily history (gate/guards/events applied); expirations/strike listings "
             "(esp. GLD/TLT far-dated) can reduce them (logged as `cadence` blocks at run time).")
    if lat_src.startswith("ASSUMED"):
        L.append("WARNING: latency is an assumption; run `probe` first for a measured number.")
    return "\n".join(L)
