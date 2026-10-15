"""Lab engine on the Theta cache path, against the offline simulator (assumed shapes; proves plumbing only)."""
import pandas as pd
import pytest

from playground.cache import DataCache
from playground.config import build_config
from playground.downloader import BudgetExceeded, Fetcher, stage_a, write_manifest
from playground.events import EventIndex
from playground.fake_theta import FakeSession
from playground.theta_client import EP
from playground.lab import run_lab
from playground.provider import SyntheticProvider
from playground.theta_client import ThetaClient
from playground.theta_provider import ThetaProvider

NOLOG = lambda *a: None


@pytest.fixture(scope="module")
def world():
    return SyntheticProvider(start="2016-01-04", end="2016-06-30", provider_greeks=False)


def mk(world, tmp_path, ticker="SPY", **kw):
    sess = FakeSession(world, only_fridays=True, **kw)
    cl = ThetaClient("http://fake", max_rps=0, session=sess, sleep=lambda s: None)
    cache = DataCache(tmp_path)
    daily = world.get_underlying_daily()
    cache.save_daily(ticker, daily)
    cache.save_daily("VIX", world.get_vix_daily().to_frame("close"))
    cache.save_daily("VIX3M", world.get_vix3m_daily().to_frame("close"))
    f = Fetcher(cl, cache, list(daily.index), threads=2, symbol=ticker)
    return sess, cl, cache, f, ThetaProvider(cache, f, ticker)


def lcfg(**kw):
    base = dict(sizing="fixed1", resolution="15m", event_buffer=False, gate=False, upside_guard=False, cb_days=0,
                entry_cadence="monthly", entry_sample="first", spot_source="daily", time_exit=21, min_credit_pct=0.01,
                start_date="2016-02-01", end_date="2016-06-30")
    base.update(kw)
    return build_config("green", base)


def test_delta_mode_uses_one_bulk_snapshot_per_expiry_and_replays_from_cache(world, tmp_path):
    sess, cl, cache, f, prov = mk(world, tmp_path)
    c = lcfg(strike_rule={"mode": "delta", "put_delta": 0.16, "call_delta": 0.16})
    r1 = run_lab(prov, c, None, "t")
    assert len(r1.trades) >= 2
    bulk = [p for path, p in sess.calls if path == "/v3/option/history/quote" and p["strike"] == "*"]
    singles_before_entry = [1 for path, p in sess.calls if path == "/v3/option/history/quote" and p["strike"] != "*"]
    assert 1 <= len(bulk) <= 2 * len(r1.trades) + 2                       # ~one bulk request per entry expiry
    assert singles_before_entry                                            # stage B legs fetched individually
    assert r1.meta["delta_source"]["provider"] > 0                         # Greeks endpoint (simulated) used
    n = len(sess.calls)
    r2 = run_lab(ThetaProvider(cache, None, "SPY"), c, None, "t")
    assert len(sess.calls) == n and r2.meta["counts"]["blocked"].get("nodata", 0) == 0
    pd.testing.assert_frame_equal(r1.trades, r2.trades)
    assert (r1.trades.spot_source == "daily").all()


def test_greeks_unsupported_falls_back_to_bs_and_is_logged(world, tmp_path):
    sess, cl, cache, f, prov = mk(world, tmp_path, greeks=False)
    c = lcfg(strike_rule={"mode": "delta", "put_delta": 0.16, "call_delta": 0.16})
    r = run_lab(prov, c, None, "t")
    assert len(r.trades) >= 2 and r.meta["delta_source"]["provider"] == 0 and r.meta["delta_source"]["bs_mid_iv"] > 0
    assert f.greeks_supported is False and cl.stats["denied"] >= 1
    assert sum(1 for path, _ in sess.calls if path == EP.OPT_GREEKS) == 1     # asked once, then stopped


def test_vix_distance_mode_confirms_cycle_expiry_with_trial(world, tmp_path):
    sess, cl, cache, f, prov = mk(world, tmp_path)
    r = run_lab(prov, lcfg(), None, "t")
    assert len(r.trades) >= 2 and all(pd.Timestamp(e).weekday() == 4 for e in r.trades.expiry)
    assert cache.trial_get(pd.Timestamp(r.trades.entry_ts.iloc[0]).date(), pd.Timestamp(r.trades.expiry.iloc[0]).date()) is True


def test_stage_a_lab_delta_mode_is_bulk_and_resumable(world, tmp_path):
    sess, cl, cache, f, prov = mk(world, tmp_path)
    c = lcfg(strike_rule={"mode": "delta", "put_delta": 0.16, "call_delta": 0.16})
    evx = EventIndex(world.synthetic_events(), list(world.get_underlying_daily().index))
    res = stage_a(prov, f, c, evx, NOLOG, threads=2, lab=True)
    assert res["done"] >= 2 and not res["errors"]
    per_strike = [1 for path, p in sess.calls if path == "/v3/option/history/quote" and p["strike"] != "*"]
    assert len(per_strike) <= 3 * res["done"]                              # trial quotes only, no strike bands
    n = len(sess.calls)
    stage_a(prov, f, c, evx, NOLOG, threads=2, lab=True)
    assert len(sess.calls) == n
    r = run_lab(ThetaProvider(cache, None, "SPY"), c, None, "t")           # entries replay from the prefetch ...
    assert r.meta["counts"]["blocked"].get("nodata", 0) == 0 and len(r.trades) >= 2     # (stage B legs come on demand)


def test_hard_stops_max_gb_and_max_hours(world, tmp_path):
    sess, cl, cache, f, prov = mk(world, tmp_path)
    f.set_limits(max_gb=1e-9)
    c = lcfg(strike_rule={"mode": "delta", "put_delta": 0.16, "call_delta": 0.16})
    evx = EventIndex(world.synthetic_events(), list(world.get_underlying_daily().index))
    res = stage_a(prov, f, c, evx, NOLOG, threads=1, lab=True)
    assert res["stopped"] and "max-gb" in res["stopped"][0] and res["done"] < res["candidates"]
    f.set_limits(max_hours=1e-12)
    with pytest.raises(BudgetExceeded):
        f.fetch_options(pd.Timestamp("2016-03-01").date(), pd.Timestamp("2016-04-01").date(), "P", [200.0], 600, 600, "15m")


def test_concurrency_never_exceeds_four(world, tmp_path):
    import threading
    import time
    live, peak = [0], [0]
    lock = threading.Lock()

    class Slow(FakeSession):
        def get(self, url, params=None, timeout=None):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.01)
            try:
                return super().get(url, params, timeout)
            finally:
                with lock:
                    live[0] -= 1
    from concurrent.futures import ThreadPoolExecutor
    cl = ThetaClient("http://fake", max_rps=0, session=Slow(world), sleep=lambda s: None)
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(lambda _: cl.expirations(), range(40)))
    assert peak[0] <= 4


def test_every_option_request_carries_the_fetchers_symbol(tmp_path):
    """Regression: multi-ticker downloads must never request SPY data for another ticker."""
    qqq = SyntheticProvider(start="2016-01-04", end="2016-06-30", ticker="QQQ", s0=100.0, vol_scale=1.25)
    sess = FakeSession(qqq, provs={"QQQ": qqq}, only_fridays=True)
    cl = ThetaClient("http://fake", max_rps=0, session=sess, sleep=lambda s: None)
    cache = DataCache(tmp_path)
    daily = qqq.get_underlying_daily()
    cache.save_daily("QQQ", daily)
    cache.save_daily("VIX", qqq.get_vix_daily().to_frame("close"))
    f = Fetcher(cl, cache, list(daily.index), threads=2, symbol="QQQ")
    prov = ThetaProvider(cache, f, "QQQ")
    for delta in (True, False):
        c = lcfg(ticker="QQQ", strike_rule={"mode": "delta", "put_delta": 0.16, "call_delta": 0.16} if delta else {"mode": "vix"})
        r = run_lab(prov, c, None, "t")
        assert len(r.trades) >= 1 and set(r.trades.ticker) == {"QQQ"}
    syms = {p["symbol"] for path, p in sess.calls if "option" in path}
    assert syms == {"QQQ"}
    # eod resolution and range fetches too
    f.fetch_options(pd.Timestamp("2016-03-01").date(), pd.Timestamp("2016-04-01").date(), "P", [100.0], 570, 960, "15m")
    assert {p["symbol"] for path, p in sess.calls} == {"QQQ"}
