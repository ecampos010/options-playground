"""Ticker configuration (profiles/tickers.yaml) and listed-strike-aware spread width."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import PROFILE_DIR, Config

DEFAULT_TICKERS = ("SPY", "QQQ", "IWM", "GLD", "TLT")


@dataclass
class TickerInfo:
    symbol: str
    width_fixed: float | None = None
    vol_index: str | None = None
    options_symbol: str | None = None
    notes: str = ""


def load_tickers(path: str | Path | None = None) -> dict[str, TickerInfo]:
    p = Path(path) if path else PROFILE_DIR / "tickers.yaml"
    raw = yaml.safe_load(p.read_text()) or {}
    return {k: TickerInfo(k, v.get("width_fixed"), v.get("vol_index"), v.get("options_symbol") or k, v.get("notes", ""))
            for k, v in raw.items()}


def increment_near(strikes: list[float], spot: float, pct: float = 0.05) -> float:
    """Local strike increment: the most common positive gap among listed strikes within +/-pct of spot."""
    near = sorted(k for k in strikes if abs(k - spot) <= pct * spot)
    gaps = [round(b - a, 4) for a, b in zip(near, near[1:]) if b > a]
    if not gaps:
        gaps = [round(b - a, 4) for a, b in zip(sorted(strikes), sorted(strikes)[1:]) if b > a] or [1.0]
    return max(set(gaps), key=gaps.count) if len(set(gaps)) > 1 else gaps[0]


def spread_width(spot: float, strikes: list[float], cfg: Config, info: TickerInfo | None = None) -> float:
    """Width in dollars: fixed (tickers.yaml / --width-mode fixed) or width_pct*spot, rounded to the nearest
    multiple of the LOCAL listed increment (at least one increment)."""
    inc = increment_near(strikes, spot)
    if cfg.width_mode == "fixed" or (cfg.width_mode == "auto" and info is not None and info.width_fixed):
        raw = info.width_fixed if (info is not None and info.width_fixed) else cfg.width
    else:
        raw = cfg.width_pct * spot
    return round(max(inc, round(raw / inc) * inc), 4)
