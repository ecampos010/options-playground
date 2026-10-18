"""SPY daily OHLC (yfinance first, then ThetaData) and VIX daily (Cboe CSV) -> data/underlying/."""
from __future__ import annotations

import io
from datetime import date
from urllib.parse import urlsplit

import pandas as pd

CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"  # TODO-VERIFY


def spy_from_yfinance(start: date, end: date, symbol: str = "SPY") -> pd.DataFrame | None:
    try:
        import yfinance as yf   # optional dependency (not in requirements.txt)
    except Exception:
        return None
    df = yf.download(symbol, start=str(start), end=str(end + pd.Timedelta(days=1)), auto_adjust=False,
                     progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]]
    df.index = pd.to_datetime(df.index).date
    return df


def vix_from_cboe(requests_mod=None) -> pd.DataFrame | None:
    import requests
    r = (requests_mod or requests).get(CBOE_VIX_URL, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Cboe VIX CSV HTTP {r.status_code}: {r.text[:200]}")
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.set_index("date")[["open", "high", "low", "close"]].sort_index()


def ensure_daily(cache, client, start: date, end: date, log, errlog=None) -> dict:
    out = {}
    spy = spy_from_yfinance(start, end)
    out["spy_source"] = "yfinance" if spy is not None else None
    if spy is None and client is not None:
        df, r = client.eod("stock", "SPY", start, end)
        if r.ok and len(df):
            spy, out["spy_source"] = df.set_index("date"), "thetadata"
        else:
            log(f"SPY daily from ThetaData failed: {r.error or r.status}")
            if errlog is not None and not r.ok:
                errlog.record(stage="daily", endpoint=urlsplit(r.url).path,
                              error_type="denied" if r.denied else "request_failed", http_status=r.status,
                              message=r.error or f"HTTP {r.status}")
    if spy is not None:
        cache.save_daily("SPY", spy)
    try:
        vix = vix_from_cboe()
        out["vix_source"] = "cboe"
    except Exception as e:
        log(f"VIX from Cboe failed ({e}); trying yfinance ^VIX")
        if errlog is not None:
            errlog.record(stage="daily", endpoint="cboe:VIX_History.csv", error_type="source_failed", message=e)
        vix, out["vix_source"] = None, None
        try:
            import yfinance as yf
            v = yf.download("^VIX", start=str(start), end=str(end + pd.Timedelta(days=1)), progress=False)
            if isinstance(v.columns, pd.MultiIndex):
                v.columns = v.columns.get_level_values(0)
            v = v.rename(columns=str.lower)[["open", "high", "low", "close"]]
            v.index = pd.to_datetime(v.index).date
            vix, out["vix_source"] = v, "yfinance"
        except Exception as e2:
            log(f"VIX from yfinance failed too: {e2}")
            if errlog is not None:
                errlog.record(stage="daily", endpoint="yfinance:^VIX", error_type="source_failed", message=e2)
    if vix is not None:
        cache.save_daily("VIX", vix)
    return out


VOL_INDICES = {"VIX3M": "VIX3M_History.csv", "VXN": "VXN_History.csv", "RVX": "RVX_History.csv",
               "GVZ": "GVZ_History.csv"}      # TODO-VERIFY URLs; skipped silently when unreachable


def ensure_vol_indices(cache, log, errlog=None) -> dict:
    """VIX3M (term-structure filter) and optional per-ticker vol indices from Cboe CSVs. UNVERIFIED."""
    import requests
    got = {}
    for name, fn in VOL_INDICES.items():
        try:
            r = requests.get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{fn}", timeout=60)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip().lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"]).dt.date
            cache.save_daily(name, df.set_index("date")[["open", "high", "low", "close"]].sort_index())
            got[name] = len(df)
        except Exception as e:
            got[name] = None
            if errlog is not None:
                errlog.record(stage="daily", endpoint=f"cboe:{fn}", error_type="source_failed", message=e)
            if name == "VIX3M":          # the others are optional: skip silently
                log(f"VIX3M unavailable ({type(e).__name__}); term-structure filter will be skipped")
    return got
