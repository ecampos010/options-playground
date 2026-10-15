"""Thermometer: four zones from editable thresholds (config/thermometer.yaml)."""
from __future__ import annotations

from pathlib import Path

import yaml

ZONE_ORDER = ["Safe", "Mild", "Danger", "Breaking point"]
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config(path=None) -> dict:
    p = Path(path) if path else CONFIG_DIR / "thermometer.yaml"
    return yaml.safe_load(p.read_text())


def zone(max_dd: float, ruin: float | None, broke: bool, cfg: dict | None = None) -> str:
    """First zone whose limits are STRICTLY satisfied. ruin=None (no Monte Carlo) -> only drawdown + broke flag."""
    cfg = cfg or load_config()
    if broke:
        return cfg.get("fallback_zone", "Breaking point")
    for z in cfg["zones"]:
        if max_dd < z["dd"] and (ruin is None or ruin < z["ruin"]):
            return z["name"]
    return cfg.get("fallback_zone", "Breaking point")


def zone_rank(name: str) -> int:
    return ZONE_ORDER.index(name) if name in ZONE_ORDER else 3
