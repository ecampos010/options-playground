"""Risk-free rate table: daily interest on idle cash and the "cash only" benchmark.

`data/rates.csv` (date,rate_pct) comes from scripts/fetch_context_data.py (FRED DFF/EFFR). Accrual is actual/360 on the
daily effective fed funds rate, the rate of a calendar day applies to the night that follows it, and a missing day
carries the last known rate forward. Days outside the file fall back to the flat `--apy` and are counted, so the
report header can say exactly how much of the window used real rates. If the file is missing the whole run uses the
flat rate and the header says so.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class RateTable:
    def __init__(self, series: pd.Series | None, flat_apy: float, source: str = "", missing_reason: str = ""):
        self.flat_apy = float(flat_apy)
        self.source = source
        self.missing_reason = missing_reason
        if series is not None and len(series):
            s = series.astype(float).sort_index()
            s = s[~s.index.duplicated(keep="last")]
            s.index = pd.DatetimeIndex(s.index).normalize()
            self.series = s
        else:
            self.series = None
        self._cache: dict = {}

    # ------------------------------------------------------------ construction
    @classmethod
    def load(cls, path: str | Path | None, flat_apy: float) -> "RateTable":
        if not path:
            return cls(None, flat_apy, missing_reason="no rates file requested")
        p = Path(path)
        if not p.exists():
            return cls(None, flat_apy, missing_reason=f"{p} not found (run scripts/fetch_context_data.py)")
        d = pd.read_csv(p)
        if not {"date", "rate_pct"} <= set(d.columns):
            raise SystemExit(f"{p}: needs columns date,rate_pct")
        d["date"] = pd.to_datetime(d["date"])
        d = d.dropna(subset=["rate_pct"])
        if d.empty:
            return cls(None, flat_apy, missing_reason=f"{p} is empty")
        return cls(d.set_index("date")["rate_pct"], flat_apy, source=str(p))

    @property
    def is_flat(self) -> bool:
        return self.series is None

    # ------------------------------------------------------------ accrual
    def _daily(self, d0: pd.Timestamp, d1: pd.Timestamp):
        """(calendar index, pct rate per calendar day, bool 'real rate' mask) over [d0, d1]."""
        cal = pd.date_range(d0, d1, freq="D")
        if self.is_flat:
            return cal, np.full(len(cal), self.flat_apy * 100.0), np.zeros(len(cal), bool)
        s = self.series
        r = s.reindex(cal.union(s.index)).ffill().reindex(cal)
        real = r.notna().to_numpy()                      # before the first observation there is no real rate
        return cal, r.fillna(self.flat_apy * 100.0).to_numpy(float), real

    def factors(self, days: pd.DatetimeIndex) -> np.ndarray:
        """factors[d] = growth of cash between days[d-1] and days[d] (factors[0] = 1)."""
        key = (days[0].value, days[-1].value, len(days), int(days.asi8.sum()))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        D = len(days)
        if self.is_flat:
            gaps = np.r_[0, np.diff(days.values).astype("timedelta64[D]").astype(int)]
            f = (1.0 + self.flat_apy) ** (gaps / 365.0)
        else:
            cal, r, _ = self._daily(days[0], days[-1])
            cum = np.r_[1.0, np.cumprod(1.0 + r / 100.0 / 360.0)]       # cum[j] = growth from cal[0] up to the start of cal[j]
            pos = cal.get_indexer(days)
            c = cum[pos]
            f = np.r_[1.0, c[1:] / c[:-1]] if D > 1 else np.ones(1)
        self._cache[key] = f
        return f

    def cash_curve(self, days: pd.DatetimeIndex, start_eq: float) -> np.ndarray:
        """Equity of an account that only holds cash and earns the risk-free rate."""
        return start_eq * np.cumprod(self.factors(days))

    def coverage(self, d0, d1) -> dict:
        """How much of [d0, d1] used real rates (vs the flat fallback) and what the file covers."""
        d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
        if self.is_flat:
            return {"source": "flat", "share_real": 0.0, "first": None, "last": None}
        _, _, real = self._daily(d0, d1)
        return {"source": self.source, "share_real": float(real.mean()) if len(real) else 0.0,
                "first": str(self.series.index[0].date()), "last": str(self.series.index[-1].date())}

    def at(self, d) -> float | None:
        """Fed funds rate (percent) in effect on date d (last known), or None if before the file / flat mode."""
        if self.is_flat:
            return None
        s = self.series[self.series.index <= pd.Timestamp(d)]
        return float(s.iloc[-1]) if len(s) else None

    def describe(self, d0=None, d1=None) -> str:
        if self.is_flat:
            why = f" ({self.missing_reason})" if self.missing_reason else ""
            return f"FLAT FALLBACK {self.flat_apy * 100:.1f}% APY on idle cash{why}: real rates not used"
        cv = self.coverage(d0, d1) if d0 is not None else {"first": str(self.series.index[0].date()), "last": str(self.series.index[-1].date()), "share_real": 1.0}
        extra = "" if cv["share_real"] >= 0.999 else f"; {100 * (1 - cv['share_real']):.0f}% of the window is before the file and uses flat {self.flat_apy * 100:.1f}%"
        return f"daily effective fed funds rate from {self.source} ({cv['first']}..{cv['last']}, actual/360){extra}"

    def regime(self, d0, d1, move_bp: float = 10.0, low: float = 0.5, high: float = 4.0) -> dict | None:
        """Level at both ends, change in basis points and a plain label for [d0, d1]."""
        if self.is_flat:
            return None
        a, b = self.at(d0), self.at(d1)
        if a is None or b is None:
            return None
        ch = (b - a) * 100.0
        trend = "hiking" if ch >= move_bp else "cutting" if ch <= -move_bp else "on hold"
        level = "near zero" if b < low else "high" if b >= high else "moderate"
        return {"start": a, "end": b, "change_bp": ch, "trend": trend, "level": level}
