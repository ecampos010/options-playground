"""Configuration, profiles (traffic-light ladder) and CLI-override plumbing.

Every Green number is a default here. Profiles (profiles/*.yaml) only override
keys of this dataclass; `tier` is metadata that picks the default budget.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

TIER_BUDGET = {"green": 0.15, "yellow": 0.05, "red": 0.01}
PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"
# Keys excluded when counting "switches" that differ between profiles.
META_KEYS = {"profile", "tier", "budget_pct"}
DICT_KEYS = {"strike_rule", "event_buffer_overrides", "event_release_times"}


@dataclass
class Config:
    profile: str = "green"
    tier: str = "green"
    # --- account ---
    start_equity: float = 100_000.0
    apy: float = 0.036
    budget_pct: float | None = None          # None -> TIER_BUDGET[tier]
    expiry_cap_frac: float = 0.60            # per-expiry cap, share of the budget
    expiry_cap: bool = True                  # switch for the per-expiry cap
    underlying_cap_frac: float = 0.50        # only enforced if >1 underlying in the run
    min_credit: float = 0.50
    width: float = 5.0
    full_use: bool = False                   # round contracts to nearest instead of floor
    commission: float = 0.0                  # $ per contract per leg (brief says nothing: 0)
    # --- entry filters ---
    gate: bool = True
    gate_vix: float = 25.0
    gate_day: float = -0.015
    gate_5d: float = -0.04
    upside_guard: bool = True
    upside_5d: float = 0.04
    event_buffer: bool = True
    event_buffer_days: int = 2               # blocks event day (offset 0) .. offset N
    event_buffer_overrides: dict = field(default_factory=dict)   # type -> days
    event_dayof_allowed: list = field(default_factory=list)      # types enterable day-of after release
    event_release_times: dict = field(default_factory=lambda: {
        "CPI": "08:30", "PPI": "08:30", "NFP": "08:30", "FOMC": "14:00"})
    event_types: list = field(default_factory=lambda: ["FOMC", "CPI", "PPI", "NFP", "ELECTION"])
    earnings_filter: bool = True
    cb_days: int = 21                        # 0 = off
    entry_requires_gate_trip: bool = False   # Red
    max_open_lots: int = 0                   # 0 = unlimited; Red = 1
    entry_every: int = 1                     # try an entry every N trading days
    # --- structure ---
    dte: list = field(default_factory=lambda: [30, 35, 45])   # min, target, max
    strike_rule: dict = field(default_factory=lambda: {
        "mode": "vix", "put_pct": 0.046, "call_pct": 0.040, "vix_ref": 15.3,
        "put_delta": 0.16, "call_delta": 0.16})
    entry_expiries: int = 3                  # expiries tried, closest to target DTE
    band_strikes: int = 10                   # +/- listed strikes around the rule strike
    # --- exits ---
    tp: float = 0.50
    stop_mult: float | None = 2.0            # None = no stop
    breach_exit: bool = True
    time_exit: int = 21                      # 0 = off
    # --- fills / entry timing ---
    fill: str = "natural"
    haircut: float = 0.05
    entry_window: str = "09:45-15:30"
    entry_sample: str = "random"
    seed: int = 1
    resolution: str = "5m"
    # --- range ---
    start_date: str | None = None
    end_date: str | None = None
    # --- hedge overlay (reserved, see reserve.py) ---
    overlay_pct: float = 0.0

    # ------------------------------------------------------------------
    def effective_budget(self) -> float:
        return self.budget_pct if self.budget_pct is not None else TIER_BUDGET.get(self.tier, 0.15)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["budget_pct"] = self.effective_budget()
        return d

    def replace(self, **kw) -> "Config":
        return apply_overrides(self, kw)

    def window(self) -> tuple[int, int]:
        a, b = self.entry_window.split("-")
        return _mins(a), _mins(b)

    def res_minutes(self) -> int:
        return 390 if self.resolution == "eod" else int(self.resolution.rstrip("m"))


def _mins(hhmm: str) -> int:
    h, m = hhmm.split(":")[:2]
    return int(h) * 60 + int(m)


def apply_overrides(cfg: Config, ov: dict) -> Config:
    names = {f.name for f in fields(Config)}
    out = Config(**{k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
                    for k, v in asdict(cfg).items()})
    for k, v in ov.items():
        if k not in names:
            raise KeyError(f"unknown config key: {k}")
        if k in DICT_KEYS and isinstance(v, dict):
            merged = dict(getattr(out, k))
            merged.update(v)
            v = merged
        setattr(out, k, v)
    return out


def parse_profile_text(text: str) -> dict:
    """Flat 'key: json-value' subset of YAML (no PyYAML dependency)."""
    out: dict = {}
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"profile line {ln}: expected 'key: value'")
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = json.loads(v.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"profile line {ln}: value must be JSON (quote strings): {e}")
    return out


def profile_names() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))


def load_profile_overrides(name: str) -> dict:
    p = PROFILE_DIR / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no profile {name!r}; available: {', '.join(profile_names())}")
    ov = parse_profile_text(p.read_text())
    ov["profile"] = name
    return ov


def build_config(profile: str = "green", overrides: dict | None = None) -> Config:
    cfg = apply_overrides(Config(), load_profile_overrides(profile))
    return apply_overrides(cfg, overrides or {})


def diff_switches(a: Config, b: Config) -> list[str]:
    da, db = asdict(a), asdict(b)
    return sorted(k for k in da if k not in META_KEYS and da[k] != db[k])
