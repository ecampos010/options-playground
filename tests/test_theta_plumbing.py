"""Plumbing tests against the OFFLINE simulator (assumed response shapes; proves nothing about ThetaData)."""
import json
from datetime import date

import pandas as pd
import pytest

from playground.cache import DataCache
from playground.config import build_config
from playground.downloader import Fetcher, candidate_days, stage_a, write_manifest
from playground.engine import run_backtest
from playground.events import EventIndex
from playground.fake_theta import FakeSession
from playground.planner import make_plan
from playground.probe import Probe
from playground.provider import MissingData, SyntheticProvider
from playground.theta_client import ThetaClient
from playground.theta_provider import ThetaProvider


@pytest.fixture(scope="module")
def world():
    return SyntheticProvider(start="2016-01-04", end="2016-09-30")


def mk(world, tmp_path, **kw):
    sess = FakeSession(world, **kw)
    cl = ThetaClient("http://fake", max_rps=0, session=sess, sleep=lambda s: None)
    cache = DataCache(tmp_path)
    daily = world.get_underlying_daily()
    cache.save_daily("SPY", daily)
    cache.save_daily("VIX", world.get_vix_daily().to_frame("close"))
    return sess, cl, cache, Fetcher(cl, cache, list(daily.index), threads=2)


CFG = dict(resolution="15m", start_date="2016-02-01", end_date="2016-05-31")


def test_download_is_idempotent_and_cache_run_matches(world, tmp_path):
    sess, cl, cache, f = mk(world, tmp_path)
    ev = world.synthetic_events()
    cfg = build_config("green", CFG)
    prov = ThetaProvider(cache, f)
    r1 = run_backtest(prov, cfg, ev, "t")
    n_calls = len(sess.calls)
    assert n_calls > 0 and len(r1.trades) > 0
    # second pass with fetching enabled: nothing new requested
    r2 = run_backtest(ThetaProvider(cache, f), cfg, ev, "t")
    assert len(sess.calls) == n_calls
    # cache-only provider reproduces the run with no gaps
    r3 = run_backtest(ThetaProvider(cache, None), cfg, ev, "t")
    assert r3.gaps == 0 and len(sess.calls) == n_calls
    pd.testing.assert_frame_equal(r2.trades, r3.trades)
    # the fetch is tight: far fewer requests than a full-chain download would need
    assert n_calls < 3000


def test_cache_only_reports_missing(world, tmp_path):
    _, _, cache, _ = mk(world, tmp_path)
    r = run_backtest(ThetaProvider(cache, None), build_config("green", CFG), world.synthetic_events(), "t")
    assert len(r.trades) == 0 and r.gaps > 0          # reported, not crashed
    with pytest.raises(MissingData):
        ThetaProvider(cache, None).get_option_quotes(date(2016, 3, 1), date(2016, 4, 1), "P", [200.0], 600, 600, "15m")


def test_stage_a_prefetch_then_engine_needs_few_extra_calls(world, tmp_path):
    sess, cl, cache, f = mk(world, tmp_path)
    cfg = build_config("green", CFG)
    ev = world.synthetic_events()
    evx = EventIndex(ev, list(world.get_underlying_daily().index))
    prov = ThetaProvider(cache, f)
    res = stage_a(prov, f, cfg, evx, lambda *a: None, threads=2, limit=15)
    assert res["done"] == 15 and not res["errors"]
    n = len(sess.calls)
    stage_a(prov, f, cfg, evx, lambda *a: None, threads=2, limit=15)     # resume: no-op
    assert len(sess.calls) == n
    m = write_manifest(cache, f, {})
    assert m["requests"] == cl.stats["requests"] and m["cache_bytes_on_disk"] > 0
    assert json.loads((cache.root / "state" / "manifest.json").read_text())["requests"] > 0


def test_bulk_entry_fetch_uses_far_fewer_requests(world, tmp_path):
    s1, c1, k1, f1 = mk(world, tmp_path / "a")
    s2, c2, k2, f2 = mk(world, tmp_path / "b")
    f2.entry_fetch = "bulk"
    cfg = build_config("green", CFG)
    evx = EventIndex(world.synthetic_events(), list(world.get_underlying_daily().index))
    stage_a(ThetaProvider(k1, f1), f1, cfg, evx, lambda *a: None, 1, limit=6)
    stage_a(ThetaProvider(k2, f2), f2, cfg, evx, lambda *a: None, 1, limit=6)
    assert len(s2.calls) * 5 < len(s1.calls)


def test_throttle_backoff_and_failures(world, tmp_path):
    sess = FakeSession(world, throttle_every=3)
    sleeps = []
    cl = ThetaClient("http://fake", max_rps=0, session=sess, sleep=sleeps.append)
    ex, r = cl.expirations()
    assert r.ok and ex and cl.stats["throttled"] >= 0
    for _ in range(6):
        cl.expirations()
    assert cl.stats["throttled"] > 0 and sleeps and sleeps[0] == 1.0
    bad = ThetaClient("http://fake", max_rps=0, max_retries=1, sleep=lambda s: None,
                      session=FakeSession(world, fail_paths=("/v3/option/list/expirations",)))
    _, r = bad.expirations()
    assert not r.ok and "500" in r.error


def test_probe_fails_soft_and_writes_report(tmp_path):
    long_world = SyntheticProvider(start="2006-01-03", end="2016-09-30")
    cl = ThetaClient("http://fake", max_rps=0, sleep=lambda s: None,
                     session=FakeSession(long_world, fail_paths=("/v3/index/history/eod",)))
    rep = Probe(cl, today=date(2016, 9, 30), out_dir=tmp_path, burst=10).run()
    assert (tmp_path / "probe_report.md").exists() and (tmp_path / "probe_report.json").exists()
    secs = {c["section"][0] for c in rep["checks"]}
    assert {"a", "b", "c", "d", "e"} <= secs
    failed = [c for c in rep["checks"] if not c["ok"]]
    assert any("index eod" in c["name"] and c["status"] == 500 and c["raw_error"] for c in failed)
    md = (tmp_path / "probe_report.md").read_text()
    assert "UNVERIFIED" in md and "raw error" in md
    assert rep["measured"].get("option_quote_latency_p50_s") is not None
    assert any(c["section"].startswith("b") and c["ok"] for c in rep["checks"])


def test_plan_prints_all_resolutions_and_writes_nothing(world, tmp_path):
    daily, vix = world.get_underlying_daily(), world.get_vix_daily()
    cfg = build_config("green", {"start_date": "2016-02-01", "end_date": "2016-09-30"})
    evx = EventIndex(world.synthetic_events(), list(daily.index))
    out = make_plan(cfg, daily, vix, evx, results_dir=tmp_path, data_dir=tmp_path)
    for res in ("1m", "5m", "15m", "eod"):
        assert res in out
    assert "wall@2thr" in out and "wall@4thr" in out and "NAIVE" in out and "ASSUMED" in out
    assert list(tmp_path.iterdir()) == []


def test_candidate_days_is_superset_of_engine_entries(world):
    cfg = build_config("green", {"resolution": "15m", "start_date": "2016-02-01", "end_date": "2016-09-30"})
    ev = world.synthetic_events()
    daily, vix = world.get_underlying_daily(), world.get_vix_daily()
    cands = set(candidate_days(cfg, daily, vix, EventIndex(ev, list(daily.index))))
    r = run_backtest(world, cfg, ev, "t")
    entered = {pd.Timestamp(x).date() for x in r.trades["entry_ts"]}
    assert entered <= cands and len(cands) < len(daily)
