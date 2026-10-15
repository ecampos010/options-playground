"""Plain-English verdicts from fixed thresholds (config/verdicts.yaml). Wording never depends on color."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .thermo import CONFIG_DIR


def load_config(path=None) -> dict:
    return yaml.safe_load(Path(path or CONFIG_DIR / "verdicts.yaml").read_text())


def _pct(x, d=0):
    return f"{abs(x) * 100:.{d}f}%"


def window_name(date: str, cfg: dict) -> str:
    for w in cfg.get("windows", []):
        if str(w["start"]) <= date <= str(w["end"]):
            return w["name"]
    return f"{date[:7]} window"


def classify(row, cfg: dict) -> str:
    if row["broke"]:
        return "Broke"
    dd = row["max_dd"]
    if dd < cfg["steady"]["max_drawdown"] and row["worst_month"] >= cfg["steady"]["worst_month"] \
            and row["worst_trade_pct"] >= cfg["steady"]["worst_trade_pct"]:
        return "Steady"
    if dd < cfg["bumpy"]["max_drawdown"]:
        return "Bumpy"
    if dd < cfg["rough"]["max_drawdown"]:
        return "Rough"
    return "Severe"


def verdict(row, cfg: dict, family_broke: dict | None = None) -> str:
    """row: dict-like of metrics. family_broke: {allocation: broke_date} for the same setup at other allocations."""
    cls = classify(row, cfg)
    parts = []
    if cls == "Broke":
        parts.append(f"Went broke on {row['broke_date']} at the {int(row['allocation'])}% allocation")
    elif cls == "Steady":
        parts.append(f"Steady: worst month -{_pct(row['worst_month'], 1)}, "
                     f"worst trade -{_pct(row['worst_trade_pct'], 1)} of equity")
    else:
        td = row.get("dd_trough_date", "")
        where = window_name(td, cfg) if td else "sample"
        parts.append(f"Survived the {where} with a {_pct(row['max_dd'])} drawdown "
                     f"(worst month -{_pct(row['worst_month'], 1)}, worst trade -{_pct(row['worst_trade_pct'], 1)} of equity)")
    if family_broke and cls != "Broke":
        worse = {a: d for a, d in family_broke.items() if a > row["allocation"]}
        if worse:
            a = min(worse)
            parts.append(f"the same setup went broke at the {int(a)}% allocation in {str(worse[a])[:4]}")
    if row.get("too_few_trades"):
        parts.append("TOO FEW TRADES - not evidence")
    s = "; ".join(parts)
    return s[0].upper() + s[1:] + "."
