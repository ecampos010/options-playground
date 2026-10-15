"""Contract selection against an offline simulator that mimics the VERIFIED Standard-account behaviour:
HTTP 472 (NO_DATA) for non-Friday expirations and for half-dollar strikes before 2020."""
from datetime import date

import pandas as pd
import pytest

from playground import rules
from playground.cache import DataCache
from playground.config import build_config
from playground.downloader import Fetcher, stage_a, write_manifest
from playground.engine import run_backtest
from playground.events import EventIndex
from playground.fake_theta import FakeSession
from playground.provider import SyntheticProvider
from playground.theta_client import ThetaClient
from playground.theta_provider import ThetaProvider

CFG = dict(resolution="15m", start_date="2016-02-01", end_date="2016-05-31")
NOLOG = lambda *a: None


@pytest.fixture(scope="module")
def world():
    return SyntheticProvider(start="2016-01-04", end="2016-09-30")


def mk(world, tmp_path, **kw):
    sess = FakeSession(world, only_fridays=True, half_strikes=True, **kw)
    cl = ThetaClient("http://fake", max_rps=0, session=sess, sleep=lambda s: None)
    cache = DataCache(tmp_path)
    daily = world.get_underlying_daily()
    cache.save_daily("SPY", daily)
    cache.save_daily("VIX", world.get_vix_daily().to_frame("close"))
    f = Fetcher(cl, cache, list(daily.index), threads=2)
    return sess, cl, cache, f, ThetaProvider(cache, f)


def test_472_is_no_data_not_error_and_not_retried():
    sleeps = []
    w = SyntheticProvider(start="2016-01-04", end="2016-02-29")
    sess = FakeSession(w, only_fridays=True)
    cl = ThetaClient("http://fake", max_rps=0, session=sess, sleep=sleeps.append)
    mon = next(e for e in w.list_expiries(date(2016, 1, 6)) if e.weekday() == 0)
    df, r = cl.option_quotes(mon, "P", 200.0, date(2016, 1, 6), date(2016, 1, 6), "5m", 600, 605)
    assert r.ok and r.no_data and r.status == 472 and len(df) == 0
    assert not sleeps and len(sess.calls) == 1                       # never retried
    assert cl.stats["no_data"] == 1 and cl.stats["errors"] == 0


def test_403_is_denied_not_retried():
    w = SyntheticProvider(start="2016-01-04", end="2016-02-29")
    sess = FakeSession(w, deny_stock=True)
    sleeps = []
    cl = ThetaClient("http://fake", max_rps=0, session=sess, sleep=sleeps.append)
    df, r = cl.eod("index", "VIX", date(2016, 1, 5), date(2016, 1, 8))
    assert not r.ok and r.denied and not sleeps and cl.stats["denied"] == 1


def test_norm_quotes_tolerates_timestamp_names_and_verified_columns():
    base = {"bid_size": [5], "bid_exchange": ["C"], "bid": [1.0], "bid_condition": [1],
            "ask_size": [7], "ask_exchange": ["C"], "ask": [1.1], "ask_condition": [1]}
    for name, val in (("timestamp", "2023-10-13T10:00:00.000"), ("quote_time", "2023-10-13 10:00:00"),
                      ("timestamp", "2023-10-13T10:00:00-04:00")):
        out = ThetaClient.norm_quotes(pd.DataFrame({name: [val], **base}), 414.0, "P")
        assert out.ts.iloc[0].hour == 10 and out.bid.iloc[0] == 1.0 and out.ask_size.iloc[0] == 7
    with pytest.raises(KeyError, match="no timestamp column"):
        ThetaClient.norm_quotes(pd.DataFrame(base))


def test_entry_expiries_fall_back_past_dead_and_non_friday_and_cache_negatives(world, tmp_path):
    sess, cl, cache, f, prov = mk(world, tmp_path)
    cfg = build_config("green", CFG)
    d = date(2016, 3, 2)
    ranked = rules.rank_expiries(world.list_expiries(d), d, cfg)
    fridays = [e for e in ranked if e.weekday() == 4]
    assert ranked[0].weekday() == 4 and len(fridays) >= 3          # Friday-first ranking
    got = prov.entry_expiries(d, cfg, 600, 200.0, 15.0)
    assert got == fridays[:3]
    n = len(sess.calls)
    assert prov.entry_expiries(d, cfg, 600, 200.0, 15.0) == got and len(sess.calls) == n   # cached
    # a Friday that answers 472 is skipped in favour of the next one; negative is cached
    sess2, cl2, cache2, f2, prov2 = mk(world, tmp_path / "b", dead_expiries=(fridays[0],))
    got2 = prov2.entry_expiries(d, cfg, 600, 200.0, 15.0)
    assert fridays[0] not in got2 and got2 == fridays[1:4]
    assert cache2.trial_get(d, fridays[0]) is False and cache2.trial_get(d, fridays[1]) is True
    assert cl2.stats["no_data"] >= 1 and cl2.stats["errors"] == 0
    n2 = len(sess2.calls)
    prov2.entry_expiries(d, cfg, 600, 200.0, 15.0)
    assert len(sess2.calls) == n2
    # cache-only provider reproduces the choice and ignores unconfirmed expiries
    assert ThetaProvider(cache2, None).entry_expiries(d, cfg, 600, 200.0, 15.0) == got2


def test_stage_a_uses_only_integer_strikes_pre_2020_and_records_no_data(world, tmp_path):
    sess, cl, cache, f, prov = mk(world, tmp_path)
    cfg = build_config("green", CFG)
    evx = EventIndex(world.synthetic_events(), list(world.get_underlying_daily().index))
    res = stage_a(prov, f, cfg, evx, NOLOG, threads=2, limit=12)
    assert res["done"] == 12 and not res["errors"]
    # half-dollar strikes are probed at most once per date, never in the bulk of the band
    frac = [p for path, p in sess.calls if path == "/v3/option/history/quote" and float(p["strike"]) != int(float(p["strike"]))]
    assert 0 < len(frac) <= 12
    got = [cache.grid_get(d) for d in world.get_underlying_daily().index if cache.grid_get(d)]
    assert got and set(got) == {"int"}
    # every stored option row is an integer strike on a Friday expiry
    files = list((tmp_path / "options").rglob("part-*.parquet"))
    assert files
    for fp in files:
        assert pd.read_parquet(fp).strike.map(lambda k: k == int(k)).all()
        exp = date.fromisoformat(fp.parts[-3].split("=")[1])
        assert exp.weekday() == 4
    m = write_manifest(cache, f, {})
    assert m["no_data_472"] > 0 and m["errors"] == 0 and m["no_data_472"] == cl.stats["no_data"]
    n = len(sess.calls)
    stage_a(prov, f, cfg, evx, NOLOG, threads=2, limit=12)         # resume: zero requests (incl. negatives)
    assert len(sess.calls) == n


def test_backtest_on_472_world_trades_only_fridays_and_replays_from_cache(world, tmp_path):
    sess, cl, cache, f, prov = mk(world, tmp_path)
    cfg = build_config("green", CFG)
    ev = world.synthetic_events()
    r1 = run_backtest(prov, cfg, ev, "t")
    assert len(r1.trades) > 0 and all(pd.Timestamp(e).weekday() == 4 for e in r1.trades.expiry)
    assert all(k == int(k) for k in r1.trades[["short_put", "long_put", "short_call", "long_call"]].to_numpy().ravel())
    n = len(sess.calls)
    r2 = run_backtest(ThetaProvider(cache, None), cfg, ev, "t")
    assert r2.gaps == 0 and len(sess.calls) == n
    pd.testing.assert_frame_equal(r1.trades, r2.trades)
    assert cl.stats["no_data"] > 0
