"""errors.csv: one row per failed request and per missing spot day, with the right fields, and the row counts match the
counters the client and the engine already keep."""
import csv
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import TI, cfg, market  # noqa: E402

from playground.cache import DataCache  # noqa: E402
from playground.downloader import Fetcher, FetchError  # noqa: E402
from playground.errorlog import COLUMNS, ErrorLog  # noqa: E402
from playground.lab import run_lab  # noqa: E402
from playground.provider import MissingData  # noqa: E402
from playground.theta_client import ApiResult  # noqa: E402

D = dt.date


class FakeClient:
    """Stands in for ThetaClient: every call returns the same failing result, and counts it the way the client does
    (403 -> stats['denied'], any other failure -> stats['errors'])."""

    def __init__(self, failure: ApiResult):
        self.failure = failure
        self.stats = {"requests": 0, "errors": 0, "denied": 0}

    def option_quotes(self, *args, **kwargs):
        self.stats["denied" if self.failure.denied else "errors"] += 1
        return pd.DataFrame(), self.failure


def _failure(status=500, error="HTTP 500: boom", denied=False):
    return ApiResult(ok=False, status=status, text="", latency=0.0, nbytes=0,
                     url="https://theta.test/v3/option/history/quote?symbol=SPY&expiration=20160318",
                     params={}, error=error, denied=denied)


def _rows(path):
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        return header, [dict(zip(header, r)) for r in reader]


def _fetcher(client, tmp_path, stage="A"):
    f = Fetcher(client, DataCache(tmp_path / "cache"), [D(2016, 3, 1), D(2016, 3, 2)], threads=1)
    f.errlog = ErrorLog(tmp_path / "errors.csv")
    f.stage = stage
    return f


def test_one_failed_request_writes_one_row_and_matches_the_client_counter(tmp_path):
    client = FakeClient(_failure())
    f = _fetcher(client, tmp_path)
    with pytest.raises(FetchError):
        f.fetch_options(D(2016, 3, 1), D(2016, 3, 18), "P", [500.0], 600, 600, "5m")
    f.errlog.close()

    header, rows = _rows(tmp_path / "errors.csv")
    assert header == COLUMNS == ["timestamp", "stage", "date", "expiry", "strike", "right", "endpoint",
                                 "error_type", "http_status", "message"]
    assert len(rows) == 1
    r = rows[0]
    assert r["stage"] == "A" and r["date"] == "2016-03-01" and r["expiry"] == "2016-03-18"
    assert r["strike"] == "500.0" and r["right"] == "P"
    assert r["endpoint"] == "/v3/option/history/quote"                    # path only, no query string
    assert r["error_type"] == "request_failed" and r["http_status"] == "500"
    assert "HTTP 500: boom" in r["message"]
    assert pd.Timestamp(r["timestamp"]).tzinfo is not None                 # UTC timestamp
    assert len(rows) == client.stats["errors"] + client.stats["denied"] == 1


def test_a_403_is_logged_as_denied_and_counted_as_denied(tmp_path):
    client = FakeClient(_failure(status=403, error="HTTP 403 (not permitted)", denied=True))
    f = _fetcher(client, tmp_path)
    with pytest.raises(FetchError):
        f.fetch_options(D(2016, 3, 1), D(2016, 3, 18), "C", [505.0], 600, 600, "5m")
    f.errlog.close()
    _, rows = _rows(tmp_path / "errors.csv")
    assert len(rows) == 1 and rows[0]["error_type"] == "denied" and rows[0]["right"] == "C"
    assert len(rows) == client.stats["errors"] + client.stats["denied"]


def test_one_missing_spot_day_writes_one_row_and_matches_the_engine_gap_count(tmp_path, monkeypatch):
    p = market()
    gap_day = p.get_underlying_daily().index[10]          # a day the lab actually iterates (p.days has warm-up)
    spot_bars = p.spot_bars

    def missing_on_one_day(d, *args, **kwargs):
        if d == gap_day:
            raise MissingData("no spot bars for this test day")
        return spot_bars(d, *args, **kwargs)

    monkeypatch.setattr(p, "spot_bars", missing_on_one_day)
    log = ErrorLog(tmp_path / "errors.csv")
    res = run_lab(p, cfg(), None, "", TI,
                  gap_log=lambda d, msg: log.record(stage="B", date=d, endpoint="spot/alpaca",
                                                    error_type="missing_spot", message=msg))
    log.close()
    _, rows = _rows(tmp_path / "errors.csv")
    gaps = [r for r in rows if r["error_type"] == "missing_spot"]
    assert len(gaps) == 1
    g = gaps[0]
    assert g["stage"] == "B" and g["date"] == str(gap_day) and g["endpoint"] == "spot/alpaca"
    assert "no spot bars" in g["message"] and g["expiry"] == "" and g["strike"] == ""
    assert len(gaps) == res.meta["spot_coverage"]["days_without_spot"] == 1


def test_no_errlog_means_no_file_and_the_same_error(tmp_path):
    client = FakeClient(_failure())
    f = Fetcher(client, DataCache(tmp_path / "cache"), [D(2016, 3, 1)], threads=1)
    with pytest.raises(FetchError, match="HTTP 500: boom"):
        f.fetch_options(D(2016, 3, 1), D(2016, 3, 18), "P", [500.0], 600, 600, "5m")
    assert f.errlog is None and not list(tmp_path.glob("**/errors.csv"))


def test_messages_are_collapsed_and_capped(tmp_path):
    log = ErrorLog(tmp_path / "errors.csv")
    log.record(error_type="request_failed", message="line one\nline   two " + "x" * 900)
    log.close()
    _, rows = _rows(tmp_path / "errors.csv")
    assert "\n" not in rows[0]["message"] and "  " not in rows[0]["message"]
    assert len(rows[0]["message"]) == 500
    assert log.summary() == {"path": str(tmp_path / "errors.csv"), "rows": 1, "by_type": {"request_failed": 1}}
