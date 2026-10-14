"""Event calendar (data/events.csv). Dates are never invented."""
from __future__ import annotations

import bisect
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/events.csv")
TYPES = {"FOMC", "CPI", "PPI", "NFP", "ELECTION", "EARNINGS"}


class EventsFileMissing(RuntimeError):
    pass


def load_events(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise EventsFileMissing(
            f"events file not found: {p}\n"
            "Create it from templates/events.csv.template (columns date,event[,symbol]) using real "
            "FOMC/CPI/PPI/NFP/Election dates, or pass --event-buffer off. Dates are never invented.")
    df = pd.read_csv(p, comment="#", dtype=str).fillna("")
    if not {"date", "event"} <= set(df.columns):
        raise ValueError(f"{p}: needs columns date,event")
    df["event"] = df["event"].str.strip().str.upper()
    bad = sorted(set(df["event"]) - TYPES)
    if bad:
        raise ValueError(f"{p}: unknown event types {bad}; allowed {sorted(TYPES)}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "symbol" not in df.columns:
        df["symbol"] = ""
    return df[["date", "event", "symbol"]].sort_values("date").reset_index(drop=True)


class EventIndex:
    """Maps each trading day to [(event_type, trading_day_offset_to_event)] and earnings dates."""

    def __init__(self, events: pd.DataFrame, trading_days: list[date], symbol: str = "SPY"):
        self.days = trading_days
        self.by_day: dict[date, list[tuple[str, int]]] = defaultdict(list)
        self.earnings: list[date] = []
        for edate, event, sym in zip(events["date"], events["event"], events["symbol"]):
            if event == "EARNINGS":
                if sym.upper() == symbol:
                    self.earnings.append(edate)
                continue
            # snap non-trading event dates to the next trading day
            j = bisect.bisect_left(trading_days, edate)
            if j >= len(trading_days):
                continue
            for off in range(0, 6):
                if j - off >= 0:
                    self.by_day[trading_days[j - off]].append((event, off))

    def upcoming(self, d: date) -> list[tuple[str, int]]:
        return self.by_day.get(d, [])

    def earnings_between(self, d0: date, d1: date) -> bool:
        return any(d0 <= e <= d1 for e in self.earnings)
