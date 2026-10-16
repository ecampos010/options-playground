"""Market context for the "why" layer: SPY and VIX daily closes, the event calendar and the fed funds rate.

Nothing here is invented: files that are missing simply leave the matching fields blank ("n/a") in the cards.
SPY / VIX come from `data/underlying/{SPY,VIX}_daily.parquet` (written by `python -m playground download --daily-only`),
else from the trade logs themselves (entry/exit spot of SPY trades; the prior-day VIX close stored at each entry), which is
sparse and labelled as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .cash import RateTable

EVENT_TYPES = ("FOMC", "CPI", "PPI", "NFP", "ELECTION")


@dataclass
class Context:
    spy: pd.Series | None = None
    vix: pd.Series | None = None
    events: pd.DataFrame | None = None        # columns date (Timestamp), event, symbol
    rates: RateTable | None = None
    spy_src: str = "not available"
    vix_src: str = "not available"
    events_src: str = "not available"
    notes: list = field(default_factory=list)

    def last_close(self, series: pd.Series | None, d) -> float | None:
        if series is None or series.empty:
            return None
        s = series[series.index <= pd.Timestamp(d)]
        return float(s.iloc[-1]) if len(s) else None

    def events_between(self, d0, d1, types=EVENT_TYPES) -> list:
        if self.events is None:
            return []
        e = self.events
        m = (e["date"] >= pd.Timestamp(d0)) & (e["date"] <= pd.Timestamp(d1)) & e["event"].isin(types)
        return [(r.date, r.event) for r in e[m].itertuples()]

    def describe(self) -> str:
        return (f"SPY: {self.spy_src}; VIX: {self.vix_src}; events: {self.events_src}; "
                f"rates: {self.rates.describe() if self.rates is not None else 'n/a'}")


def read_events(path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", dtype=str).fillna("")
    if not {"date", "event"} <= set(df.columns):
        raise SystemExit(f"{path}: needs columns date,event[,symbol]")
    df["event"] = df["event"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"])
    if "symbol" not in df.columns:
        df["symbol"] = ""
    return df[["date", "event", "symbol"]].sort_values(["date", "event"]).reset_index(drop=True)


def _daily_close(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    d = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    cols = {c.lower(): c for c in d.columns}
    dc = cols.get("date") or d.columns[0]
    pc = cols.get("close") or cols.get("adj close") or cols.get("adj_close")
    if pc is None:
        return None
    s = pd.Series(d[pc].astype(float).to_numpy(), index=pd.to_datetime(d[dc])).dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def spy_from_csv(path) -> pd.Series | None:
    return _daily_close(Path(path)) if path and Path(path).exists() else None


def from_trade_logs(trades: pd.DataFrame):
    """Sparse SPY closes and VIX closes recovered from the logs (used only when the daily files are absent)."""
    spy = vix = None
    s = trades[trades["ticker"] == "SPY"]
    if len(s):
        pts = pd.concat([pd.Series(s["entry_spot"].to_numpy(float), index=s["entry_ts"].dt.normalize()),
                         pd.Series(s["exit_spot"].to_numpy(float), index=s["exit_ts"].dt.normalize())])
        spy = pts.groupby(level=0).mean().sort_index()
    v = trades.dropna(subset=["entry_vix"])
    if len(v):
        idx = v["entry_ts"].dt.normalize() - pd.offsets.BDay(1)         # entry_vix is the previous trading day's close
        vix = pd.Series(v["entry_vix"].to_numpy(float), index=idx).groupby(level=0).mean().sort_index()
    return spy, vix


def load_context(data_dir, events_path, rates: RateTable, trades: pd.DataFrame | None = None, spy_csv=None) -> Context:
    ctx = Context(rates=rates)
    dd = Path(data_dir) if data_dir else None
    if dd is not None:
        ctx.spy = _daily_close(dd / "underlying" / "SPY_daily.parquet")
        ctx.vix = _daily_close(dd / "underlying" / "VIX_daily.parquet")
        if ctx.spy is not None:
            ctx.spy_src = str(dd / "underlying" / "SPY_daily.parquet")
        if ctx.vix is not None:
            ctx.vix_src = str(dd / "underlying" / "VIX_daily.parquet")
    if ctx.spy is None and spy_csv:
        ctx.spy = spy_from_csv(spy_csv)
        if ctx.spy is not None:
            ctx.spy_src = str(spy_csv)
    if trades is not None and (ctx.spy is None or ctx.vix is None):
        spy, vix = from_trade_logs(trades)
        if ctx.spy is None and spy is not None:
            ctx.spy, ctx.spy_src = spy, "trade logs (sparse: SPY trade entries/exits only)"
        if ctx.vix is None and vix is not None:
            ctx.vix, ctx.vix_src = vix, "trade logs (sparse: VIX at trade entries only)"
    p = Path(events_path) if events_path else None
    if p is not None and p.exists():
        ctx.events = read_events(p)
        ctx.events_src = str(p)
    else:
        ctx.notes.append(f"events file {events_path} not found: event lines are n/a (run scripts/fetch_context_data.py)")
    return ctx
