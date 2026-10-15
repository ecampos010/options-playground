"""`stress`: replay a strategy/profile through approximate stress windows with FORCED entries (unit mode)."""
from __future__ import annotations

import csv

import pandas as pd
from datetime import date, timedelta
from pathlib import Path

from . import factory
from .config import apply_overrides, build_config
from .lab import run_id_for, run_lab, save_lab_run

DEFAULT_PATH = Path("data/stress_windows.csv")


def load_windows(path=DEFAULT_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; it ships with the repo (data/stress_windows.csv) - edit as needed")
    lines = [ln for ln in p.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def run_stress(args_ns, windows: list[dict], profile: str, overrides: dict, every_day: bool = False,
               runs_dir="runs", command="", log=print) -> list[dict]:
    out = []
    for w in windows:
        start, end = date.fromisoformat(w["start"]), date.fromisoformat(w["end"])
        cfg = build_config(profile, overrides)
        # let positions run past the window; entries are confined to it (cfg window)
        cfg = apply_overrides(cfg, {"sizing": "fixed1", "force_entry": True, "entry_cadence": "daily",
                                    "start_date": str(start), "end_date": str(end + timedelta(days=cfg.dte[2] + 15)),
                                    "profile": f"stress_{w['name']}"})
        prov = factory.make(args_ns.provider, cfg.ticker, args_ns.data_dir, args_ns.syn_start, args_ns.syn_end,
                            args_ns.syn_seed, rate=cfg.apy)
        win = {"name": w["name"], "start": w["start"], "end": w["end"], "approximate": True}
        rid = run_id_for(cfg, {"ticker": cfg.ticker, "events": "off(forced)", "window": win})
        if (Path(runs_dir) / rid / "meta.json").exists():           # resumable
            t = pd.read_csv(Path(runs_dir) / rid / "trades.csv")
            p = Path(runs_dir) / rid
        else:
            res = run_lab(prov, cfg, None, "off(forced)", window=win, max_entries=None if every_day else 1)
            p, t = save_lab_run(res, runs_dir, command), res.trades
        out.append({"window": w["name"], "run_id": rid, "trades": len(t),
                    "pnl_total": float(t.pnl_per_contract.sum()) if len(t) else 0.0,
                    "worst": float(t.pnl_per_contract.min()) if len(t) else 0.0, "path": str(p)})
        log(f"stress {w['name']}: {out[-1]['trades']} trade(s), total {out[-1]['pnl_total']:.0f}$/contract")
    return out
