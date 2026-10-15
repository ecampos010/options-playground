"""Parquet cache under data/, partitioned by date/expiry. Tracks *coverage* so re-runs never refetch."""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

OPT_COLS = ["ts", "strike", "bid", "ask", "bid_size", "ask_size"]
K = lambda x: round(float(x), 3)


class DataCache:
    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._cov: dict[str, list] = {}
        self._data: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------ small helpers
    def _atomic_write_text(self, p: Path, text: str):
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(text)
        os.replace(tmp, p)

    def _write_parquet(self, p: Path, df: pd.DataFrame):
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{uuid.uuid4().hex[:8]}.tmp")
        df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, p)

    # ------------------------------------------------------------ daily
    def save_daily(self, name: str, df: pd.DataFrame):
        d = df.copy()
        d.index = pd.to_datetime(d.index)
        self._write_parquet(self.root / "underlying" / f"{name}_daily.parquet", d.reset_index(names="date"))

    def load_daily(self, name: str) -> pd.DataFrame | None:
        p = self.root / "underlying" / f"{name}_daily.parquet"
        if not p.exists():
            return None
        d = pd.read_parquet(p)
        d["date"] = pd.to_datetime(d["date"]).dt.date
        return d.set_index("date").sort_index()

    # ------------------------------------------------------------ listings
    def save_expirations(self, exps: list[date]):
        self._atomic_write_text(self.root / "listing" / "expirations.json", json.dumps([str(e) for e in exps]))

    def expirations(self) -> list[date] | None:
        p = self.root / "listing" / "expirations.json"
        return [date.fromisoformat(x) for x in json.loads(p.read_text())] if p.exists() else None

    def save_strikes(self, expiry: date, strikes: list[float]):
        self._atomic_write_text(self.root / "listing" / "strikes" / f"{expiry}.json", json.dumps(strikes))

    def strikes(self, expiry: date) -> list[float] | None:
        p = self.root / "listing" / "strikes" / f"{expiry}.json"
        return json.loads(p.read_text()) if p.exists() else None

    # ------------------------------------------------------------ contract-selection memo
    # Per trade date: {"<expiry>": true/false (trial quote returned rows?), "_grid": "int"|"half"}.
    def _trial_path(self, d):
        return self.root / "listing" / "trial" / f"date={d}.json"

    def _trial_map(self, d) -> dict:
        key = f"trial:{d}"
        if key not in self._cov:
            p = self._trial_path(d)
            self._cov[key] = json.loads(p.read_text()) if p.exists() else {}
        return self._cov[key]

    def trial_get(self, d, expiry) -> bool | None:
        with self._lock:
            return self._trial_map(d).get(str(expiry))

    def trial_set(self, d, expiry, listed: bool):
        with self._lock:
            m = self._trial_map(d)
            m[str(expiry)] = bool(listed)
            self._atomic_write_text(self._trial_path(d), json.dumps(m))

    def grid_get(self, d) -> str | None:
        with self._lock:
            return self._trial_map(d).get("_grid")

    def grid_set(self, d, grid: str):
        with self._lock:
            m = self._trial_map(d)
            m["_grid"] = grid
            self._atomic_write_text(self._trial_path(d), json.dumps(m))

    def bulk_done(self, res, d, expiry, m) -> bool:
        with self._lock:
            return bool(self._trial_map(d).get(f"bulk|{res}|{expiry}|{m}"))

    def bulk_set(self, res, d, expiry, m):
        with self._lock:
            mp = self._trial_map(d)
            mp[f"bulk|{res}|{expiry}|{m}"] = True
            self._atomic_write_text(self._trial_path(d), json.dumps(mp))

    def greeks_put(self, res, d, expiry, m, df: pd.DataFrame | None):
        """df None/empty = known unsupported/no data (negative cache)."""
        p = self.root / "greeks" / f"res={res}" / f"date={d}" / f"expiry={expiry}" / f"m={m}.parquet"
        d0 = df if df is not None else pd.DataFrame({"right": pd.Series([], dtype=object), "strike": pd.Series([], dtype=float),
                                                      "delta": pd.Series([], dtype=float), "iv": pd.Series([], dtype=float),
                                                      "underlying_price": pd.Series([], dtype=float)})
        self._write_parquet(p, d0)

    def greeks_get(self, res, d, expiry, m) -> pd.DataFrame | None:
        p = self.root / "greeks" / f"res={res}" / f"date={d}" / f"expiry={expiry}" / f"m={m}.parquet"
        return pd.read_parquet(p) if p.exists() else None

    # ------------------------------------------------------------ underlying bars
    def _bars_path(self, res, d):
        return self.root / "underlying_bars" / f"res={res}" / f"date={d}.parquet"

    def has_bars(self, res, d) -> bool:
        return self._bars_path(res, d).exists()

    def write_bars(self, res, d, df: pd.DataFrame):
        self._write_parquet(self._bars_path(res, d), df[["ts", "spot"]].reset_index(drop=True))

    def read_bars(self, res, d) -> pd.DataFrame:
        df = pd.read_parquet(self._bars_path(res, d))
        return df.set_index("ts").sort_index()

    # ------------------------------------------------------------ options
    def _dir(self, res, d, expiry, right) -> Path:
        return self.root / "options" / f"res={res}" / f"date={d}" / f"expiry={expiry}" / f"right={right}"

    def _coverage(self, dr: Path) -> list:
        key = str(dr)
        if key not in self._cov:
            f = dr / "_coverage.json"
            raw = json.loads(f.read_text()) if f.exists() else []
            self._cov[key] = [(frozenset(K(s) for s in e["strikes"]), e["t0"], e["t1"]) for e in raw]
        return self._cov[key]

    def covered(self, res, d, expiry, right, strike, t0, t1) -> bool:
        k = K(strike)
        with self._lock:
            return any(k in s and a <= t0 and b >= t1 for s, a, b in self._coverage(self._dir(res, d, expiry, right)))

    def write_options(self, res, d, expiry, right, requested_strikes, t0, t1, df: pd.DataFrame):
        """Persist rows for one (date, expiry, right) and record that `requested_strikes` x [t0,t1] are covered
        (also when df has no rows for them: 'known empty')."""
        dr = self._dir(res, d, expiry, right)
        with self._lock:
            cov_f = dr / "_coverage.json"
            raw = json.loads(cov_f.read_text()) if cov_f.exists() else []
            part = None
            if len(df):
                n = len(list(dr.glob("part-*.parquet"))) if dr.exists() else 0
                part = f"part-{n:05d}.parquet"
                self._write_parquet(dr / part, df[OPT_COLS].reset_index(drop=True))
            raw.append({"strikes": sorted(K(s) for s in requested_strikes), "t0": t0, "t1": t1, "file": part})
            self._atomic_write_text(cov_f, json.dumps(raw))
            self._cov.pop(str(dr), None)
            self._data.pop(str(dr), None)

    def read_options(self, res, d, expiry, right, strikes, t0, t1) -> pd.DataFrame:
        dr = self._dir(res, d, expiry, right)
        with self._lock:
            key = str(dr)
            if key not in self._data:
                parts = sorted(dr.glob("part-*.parquet")) if dr.exists() else []
                df = (pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
                      if parts else pd.DataFrame(columns=OPT_COLS))
                if df.empty:     # known-empty (NO_DATA) partitions
                    df = pd.DataFrame({"ts": pd.Series([], dtype="datetime64[ns]"),
                                       **{c: pd.Series([], dtype=float) for c in OPT_COLS[1:]}})
                df = df.drop_duplicates(["ts", "strike"]).sort_values(["strike", "ts"])
                df["_m"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
                if len(self._data) > 3000:
                    self._data.clear()
                self._data[key] = df
            df = self._data[key]
        ks = {K(s) for s in strikes}
        m = df["strike"].round(3).isin(ks) & (df["_m"] >= t0) & (df["_m"] <= t1)
        return df.loc[m, OPT_COLS].reset_index(drop=True)

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file()) if self.root.exists() else 0
