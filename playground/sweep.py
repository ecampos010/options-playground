"""`sweep`: resumable parameter grid -> runs/<run_id>/ (unit mode). Never prints per-trade output."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import factory
from .config import build_config
from .events import load_events
from .lab import run_id_for, run_lab, save_lab_run
from .report import table

PROFILE_KEY = "profile"


def grid_overrides(key: str, val) -> dict:
    """Map a grid key/value to config overrides. Dict values are merged as-is (named bundles)."""
    if isinstance(val, dict):
        return val
    if key == "delta":
        return {"strike_rule": {"mode": "delta", "put_delta": float(val), "call_delta": float(val)}}
    if key == "put_delta":
        return {"strike_rule": {"mode": "delta", "put_delta": float(val)}}
    if key == "call_delta":
        return {"strike_rule": {"mode": "delta", "call_delta": float(val)}}
    if key == "strike_mode":
        return {"strike_rule": {"mode": "vix" if val in ("vix", "vix-distance") else val}}
    if key == "stop":
        return {"stop_mult": None if val in (None, "none") else float(val)}
    if key == "dte":
        return {"dte_target": int(val)}
    if key == "trigger_action":
        return {}
    return {key: val}


def expand(spec: dict) -> list[tuple[str, dict]]:
    """-> [(profile, overrides)]: the explicit `runs` list FIRST (put the baseline there), then every grid combination."""
    base = dict(spec.get("base") or {})
    grid = spec.get("grid") or {}
    keys = list(grid)
    combos = []
    for r in spec.get("runs") or []:
        r = dict(r)
        prof = r.pop(PROFILE_KEY, base.get(PROFILE_KEY, "green"))
        ov = {}
        for k, v in r.items():
            for kk, vv in grid_overrides(k, v).items():
                ov[kk] = {**ov.get(kk, {}), **vv} if kk == "strike_rule" and kk in ov else vv
        combos.append((prof, ov))
    for vals in itertools.product(*[grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]) if keys else []:
        ov, prof = {}, base.get(PROFILE_KEY, "green")
        for k, v in zip(keys, vals):
            if k == PROFILE_KEY:
                prof = v
                continue
            for kk, vv in grid_overrides(k, v).items():
                ov[kk] = {**ov[kk], **vv} if kk == "strike_rule" and kk in ov else vv
        combos.append((prof, ov))
    if not combos:
        combos = [(base.get(PROFILE_KEY, "green"), {})]
    out = []
    for prof, ov in combos:
        b = {kk: vv for k, v in base.items() if k != PROFILE_KEY for kk, vv in grid_overrides(k, v).items()}
        sr = {**b.get("strike_rule", {}), **ov.get("strike_rule", {})}
        merged = {**b, **ov}
        if sr:
            merged["strike_rule"] = sr
        merged["sizing"] = "fixed1"
        out.append((prof, merged))
    return out


def curve_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n": 0, "win": np.nan, "avg": np.nan, "total": 0.0, "dd": 0.0, "ror": np.nan, "h1": 0.0, "h2": 0.0}
    t = t.sort_values("exit_ts")
    c = t["pnl_per_contract"].cumsum()
    dd = float((c - c.cummax()).min())
    mid = pd.to_datetime(t["exit_ts"]).sort_values().iloc[len(t) // 2]
    ex = pd.to_datetime(t["exit_ts"])
    def ror(m):
        c = t.loc[m, "collateral_per_contract"].sum()
        return float(t.loc[m, "pnl_per_contract"].sum() / c) if c else 0.0
    return {"n": len(t), "win": float((t.pnl_per_contract > 0).mean()), "avg": float(t.pnl_per_contract.mean()),
            "total": float(t.pnl_per_contract.sum()), "dd": dd,
            "ror": float(t.pnl_per_contract.sum() / t.collateral_per_contract.sum()),
            "h1": ror(ex < mid), "h2": ror(ex >= mid)}


def run_stress_sweep(spec: dict, args_ns, runs_dir, log, command) -> dict:
    from .stress import load_windows, run_stress
    ws = load_windows(spec.get("windows", "data/stress_windows.csv"))
    combos = expand(spec)
    rows = []
    for n, (prof, ov) in enumerate(combos, 1):
        for r in run_stress(args_ns, ws, prof, ov, every_day=bool(spec.get("every_day")), runs_dir=runs_dir,
                            command=command, log=lambda *a: None):
            rows.append({"idx": len(rows) + 1, "run_id": r["run_id"], "ticker": ov.get("ticker", "SPY"),
                         "strategy": ov.get("strategy", "iron_condor"), "profile": prof, "tag": r["window"],
                         "n": r["trades"], "win": float("nan"), "avg": r["pnl_total"] / max(r["trades"], 1),
                         "total": r["pnl_total"], "dd": r["worst"], "ror": float("nan"), "h1": 0.0, "h2": 0.0})
        log(f"[{n}/{len(combos)}] stress combo done")
    df = pd.DataFrame(rows)
    return {"name": spec.get("name", "stress"), "tried": len(rows), "ran": len(rows), "resumed_skipped": 0,
            "summary": df, "stress": True}


def run_sweep(spec: dict, args_ns, runs_dir="runs", log=print, command="") -> dict:
    name = spec.get("name", "sweep")
    if spec.get("type") == "stress":
        return run_stress_sweep(spec, args_ns, runs_dir, log, command)
    combos = expand(spec)
    if getattr(args_ns, "limit", None):
        combos = combos[: args_ns.limit]
    total = len(combos)
    rows, done, skipped = [], 0, 0
    events_cache: dict = {}
    for n, (prof, ov) in enumerate(combos, 1):
        cfg = build_config(prof, ov)
        provider = factory.make(args_ns.provider, cfg.ticker, args_ns.data_dir, args_ns.syn_start, args_ns.syn_end,
                                args_ns.syn_seed, rate=cfg.apy)
        if cfg.event_buffer or cfg.earnings_filter:
            key = cfg.ticker
            if key not in events_cache:
                p = Path(args_ns.events)
                if p.exists():
                    events_cache[key] = (load_events(p), str(p))
                elif provider.name == "synthetic":
                    events_cache[key] = (provider.synthetic_events(), "SYNTHETIC-calendar(not real dates)")
                else:
                    from .events import EventsFileMissing
                    raise EventsFileMissing(f"events file not found: {p}; create it from templates/events.csv.template")
            ev, note = events_cache[key]
        else:
            ev, note = None, "off"
        rid = run_id_for(cfg, {"ticker": cfg.ticker, "events": note, "window": None})
        out = Path(runs_dir) / rid
        if (out / "meta.json").exists():
            skipped += 1
            t = pd.read_csv(out / "trades.csv")
            mj = json.loads((out / "meta.json").read_text())
            tag, blocked = mj.get("config_tag", ""), mj.get("counts", {}).get("blocked", {})
        else:
            res = run_lab(provider, cfg, ev, note)
            save_lab_run(res, runs_dir, command)
            t, tag, blocked = res.trades, res.config_tag, res.meta["counts"]["blocked"]
            done += 1
        st = curve_stats(t)
        rows.append({"idx": n, "run_id": rid, "ticker": cfg.ticker, "strategy": cfg.strategy, "profile": prof,
                     "tag": tag, "top_block": max(blocked, key=blocked.get) if blocked else "", **st})
        if n % max(1, total // 25) == 0 or n == total:
            log(f"[{n}/{total}] {cfg.ticker} {cfg.strategy} {tag}: trades={st['n']} avg={st['avg']:.2f} total={st['total']:.0f}")
    df = pd.DataFrame(rows)
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(runs_dir) / f"_sweep_{name}_summary.csv", index=False)
    return {"name": name, "tried": total, "ran": done, "resumed_skipped": skipped, "summary": df}


def format_summary(res: dict, top: int = 12) -> str:
    df, tried = res["summary"], res["tried"]
    L = [f"SWEEP {res['name']}: CONFIGURATIONS TRIED = {tried} (ran {res['ran']}, resumed/skipped {res['resumed_skipped']})",
         "Multiple-comparison warning: with this many configurations, the best ones are partly luck; "
         "judge by the half-vs-half columns, not the ranking."]
    if df.empty:
        return "\n".join(L)
    if res.get("stress"):
        L.append(table([[r.tag, r.ticker + "/" + r.strategy, r.profile, r.n, f"{r.total:.0f}", f"{r.dd:.0f}"]
                        for r in df.itertuples()], ["window", "ticker/strategy", "profile", "trades", "total$/c", "worst$/c"]))
        return "\n".join(L)
    base = df.iloc[0]
    empty = df[df.n == 0]
    if len(empty):
        why = empty.top_block.value_counts().to_dict()
        L.append(f"{len(empty)} configuration(s) produced NO trades (excluded from the ranking; main blocker: {why}) - "
                 f"e.g. the 10%-of-width minimum credit rejects far-OTM deltas.")
    df = df[df.n > 0]
    rank = df.sort_values("ror", ascending=False).head(top)
    rows = []
    for r in rank.itertuples():
        better = r.ror > base.ror and r.idx != base.idx
        rows.append([r.idx, f"{r.ticker}/{r.strategy}", r.tag[:44], r.n, f"{r.win:.0%}" if r.n else "-",
                     f"{r.avg:.1f}", f"{100 * r.ror:.1f}%", f"{r.dd:.0f}",
                     f"{100 * r.h1:.1f}%/{100 * r.h2:.1f}%" + ("  <- better than baseline" if better else "")])
    L.append(table(rows, ["#", "ticker/strategy", "config", "trades", "win", "avg$/c", "return on risk", "maxDD$/c",
                          "ror H1/H2"]))
    L.append(f"baseline = #{int(base.idx)} {base.tag}: return on risk {100 * base.ror:.1f}% (first half {100 * base.h1:.1f}%, "
             f"second half {100 * base.h2:.1f}%), {int(base.n)} trades")
    bet = df[(df.ror > base.ror) & (df.idx != base.idx)]
    if len(bet):
        same = bet[(bet.h1 > base.h1) & (bet.h2 > base.h2)]
        L.append(f"{len(bet)} of {tried - 1} beat the baseline's return on risk; only {len(same)} also beat it in BOTH halves "
                 f"(baseline {100 * base.h1:.1f}% / {100 * base.h2:.1f}%). Few trades per config => noisy; do not trust small n.")
    return "\n".join(L)


def load_spec(path) -> dict:
    return yaml.safe_load(Path(path).read_text())
