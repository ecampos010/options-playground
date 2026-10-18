"""Per-run errors.csv for the lab downloader: one row per failed request, per missing spot day, per Stage A day that
could not be covered, and per failed daily-data source.

Write-only. The downloader never reads the file back, so a row cannot change what is fetched, cached or retried.

Columns (in this order):
  timestamp    UTC, ISO 8601, to the second, when the row was written
  stage        listing | daily | A | B | greeks (who made the request; see the stage names in cli_lab)
  date         trading day the request was for (for a chunk request: the chunk's first day); blank if none
  expiry       option expiry; blank if none
  strike       strike; blank if none
  right        P or C; blank if none
  endpoint     URL path of the request (no query string), or a source name such as cboe:VIX_History.csv
  error_type   request_failed | denied | missing_spot | day_error | source_failed
               request_failed: HTTP status other than 200/472/403, or a connection error (http_status -1)
               denied:         HTTP 403, not included in the subscription
               missing_spot:   the lab engine had no spot bars for that day
               day_error:      Stage A could not cover that entry day (message is its reason)
               source_failed:  a daily source outside the ThetaData client failed (Cboe, yfinance)
  http_status  HTTP status when there is one, else blank
  message      the error text, whitespace-collapsed and cut to 500 characters

Counter relation: each failed ThetaData request is counted once by the client in stats["errors"] or stats["denied"],
so rows with error_type request_failed or denied equal those two counters for requests made through the Fetcher.
"""
from __future__ import annotations

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

COLUMNS = ["timestamp", "stage", "date", "expiry", "strike", "right", "endpoint", "error_type", "http_status", "message"]
MAX_MESSAGE = 500


def _s(x) -> str:
    return "" if x is None else str(x)


class ErrorLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(COLUMNS)
        self._fh.flush()
        self._lock = threading.Lock()                 # Stage A and the strike threads write concurrently
        self.counts: dict[str, int] = {}

    def record(self, *, error_type: str, stage: str = "", date=None, expiry=None, strike=None, right: str = "",
               endpoint: str = "", http_status=None, message: str = "") -> None:
        row = [datetime.now(timezone.utc).isoformat(timespec="seconds"), stage, _s(date), _s(expiry),
               _s(strike), right or "", endpoint or "", error_type, _s(http_status),
               " ".join(str(message).split())[:MAX_MESSAGE]]
        with self._lock:
            self._w.writerow(row)
            self._fh.flush()
            self.counts[error_type] = self.counts.get(error_type, 0) + 1

    def summary(self) -> dict:
        with self._lock:
            return {"path": str(self.path), "rows": sum(self.counts.values()), "by_type": dict(self.counts)}

    def close(self) -> None:
        with self._lock:
            self._fh.close()
