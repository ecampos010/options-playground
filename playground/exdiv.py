"""Ex-dividend dates (for early-assignment flags). Alpaca corporate actions -> yfinance -> skip with a warning.
UNVERIFIED until run on the VPS. Cached as data/spot/<T>/exdiv.csv (not vendor option data)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def exdiv_path(root, ticker) -> Path:
    return Path(root) / ticker / "exdiv.csv"


def load_exdiv(root, ticker) -> list[date]:
    if root is None:
        return []
    p = exdiv_path(root, ticker)
    if not p.exists():
        return []
    return sorted(pd.to_datetime(pd.read_csv(p)["ex_date"]).dt.date)


def fetch_exdiv(ticker: str, start: date, end: date, alpaca=None, log=print) -> tuple[list[date], str]:
    """Returns (dates, source). source 'none' (with a warning) when nothing is reachable."""
    if alpaca is not None:
        try:   # TODO-VERIFY: /v1/corporate-actions?symbols=..&types=cash_dividend&start=..&end=..
            js = alpaca._get("/v1/corporate-actions", {"symbols": ticker, "types": "cash_dividend",
                                                       "start": str(start), "end": str(end), "limit": 1000})
            rows = (js.get("corporate_actions") or {}).get("cash_dividends", [])
            ds = sorted({pd.to_datetime(r["ex_date"]).date() for r in rows if r.get("ex_date")})
            if ds:
                return ds, "alpaca"
        except Exception as e:
            log(f"exdiv {ticker}: Alpaca corporate actions failed ({type(e).__name__}); trying yfinance")
    try:
        import yfinance as yf
        s = yf.Ticker(ticker).dividends
        ds = sorted({pd.Timestamp(x).tz_localize(None).date() for x in s.index if start <= pd.Timestamp(x).date() <= end})
        if ds:
            return ds, "yfinance"
    except Exception as e:
        log(f"exdiv {ticker}: yfinance unavailable ({type(e).__name__})")
    log(f"WARNING: no ex-dividend dates for {ticker}; exdiv_flag will be 0 for everything (data unavailable)")
    return [], "none"


def save_exdiv(root, ticker, dates: list[date]):
    p = exdiv_path(root, ticker)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ex_date": [str(d) for d in dates]}).to_csv(p, index=False)
