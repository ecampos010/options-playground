"""Metrics and compact text summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import REASONS, Result


def metrics(res: Result) -> dict:
    eq = res.equity
    t = res.trades
    start = res.cfg.start_equity
    if eq.empty:
        return {"final": start, "cagr": 0.0, "maxdd": 0.0, "trades": 0, "win": float("nan"),
                "avg_credit": float("nan"), "years": 0.0}
    final = float(eq["equity"].iloc[-1])
    d0, d1 = pd.Timestamp(eq["date"].iloc[0]), pd.Timestamp(eq["date"].iloc[-1])
    years = max((d1 - d0).days / 365.25, 1e-9)
    cagr = (final / start) ** (1 / years) - 1 if final > 0 else -1.0
    curve = np.concatenate([[start], eq["equity"].to_numpy(float)])
    dd = curve / np.maximum.accumulate(curve) - 1
    n = len(t)
    return {"final": final, "cagr": cagr, "maxdd": float(dd.min()), "trades": n,
            "win": float((t["pnl"] > 0).mean()) if n else float("nan"),
            "avg_credit": float(t["credit"].mean()) if n else float("nan"), "years": years}


def exit_counts(res: Result) -> dict:
    t = res.trades
    return {r: int((t["exit_reason"] == r).sum()) if len(t) else 0 for r in REASONS}


def year_end(res: Result) -> list[tuple[int, float]]:
    eq = res.equity
    if eq.empty:
        return []
    e = eq.assign(y=pd.to_datetime(eq["date"]).dt.year).groupby("y")["equity"].last()
    return [(int(y), float(v)) for y, v in e.items()]


def config_line(res: Result) -> str:
    c = res.cfg
    stop = f"{c.stop_mult}x" if c.stop_mult is not None else "off"
    return (f"{c.profile.upper()} | data={res.provider} | fill={c.fill}"
            f"{'(' + str(c.haircut) + ')' if c.fill == 'haircut' else ''} | seed={c.seed} | "
            f"entry={c.entry_window}/{c.entry_sample} | res={c.resolution} | budget={c.effective_budget():.0%} | "
            f"gate={'on' if c.gate else 'off'} upside={'on' if c.upside_guard else 'off'} "
            f"event={'on' if c.event_buffer else 'off'} cb={c.cb_days} | tp={c.tp} stop={stop} "
            f"time={c.time_exit} dte={c.dte} | strikes={c.strike_rule['mode']} | events={res.events_note}")


def compact_summary(res: Result) -> str:
    m = metrics(res)
    ex = exit_counts(res)
    b = res.blocked
    L = [config_line(res)]
    L.append(f"final equity {m['final']:,.0f} (start {res.cfg.start_equity:,.0f}) | CAGR {m['cagr']:.1%} | "
             f"maxDD {m['maxdd']:.1%} | {m['years']:.1f} yrs")
    L.append(f"trades {m['trades']} | win rate {m['win']:.1%} | avg credit {m['avg_credit']:.2f}")
    L.append("exits: " + " ".join(f"{k}={v}" for k, v in ex.items()))
    L.append("blocked days: " + " ".join(f"{k}={b.get(k, 0)}" for k in ("gate", "upside", "event", "cb")) +
             " | other: " + " ".join(f"{k}={v}" for k, v in sorted(b.items())
                                      if k not in ("gate", "upside", "event", "cb")) +
             (f" | DATA GAPS={res.gaps}" if res.gaps else ""))
    L.append("year-end equity: " + "  ".join(f"{y}:{v:,.0f}" for y, v in year_end(res)))
    return "\n".join(L[:40])


def save_run(res: Result, root: str | Path = "results", run_id: str | None = None) -> Path:
    import hashlib
    cfgd = res.cfg.to_dict()
    run_id = run_id or f"{res.cfg.profile}_{res.cfg.fill}_{hashlib.md5(json.dumps(cfgd, sort_keys=True, default=str).encode()).hexdigest()[:6]}"
    out = Path(root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / "trades.csv", index=False)
    res.equity.to_csv(out / "equity.csv", index=False)
    (out / "config.json").write_text(json.dumps({"config": cfgd, "provider": res.provider,
                                                 "events": res.events_note, "blocked": res.blocked,
                                                 "data_gaps": res.gaps}, indent=2, default=str))
    (out / "summary.md").write_text("```\n" + compact_summary(res) + "\n```\n")
    return out


def table(rows: list[list], header: list[str]) -> str:
    w = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = " | ".join("{:<%d}" % n for n in w)
    return "\n".join([fmt.format(*header), "-+-".join("-" * n for n in w)] + [fmt.format(*map(str, r)) for r in rows])
