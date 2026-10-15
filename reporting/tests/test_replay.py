import numpy as np
import pytest

from reporting import grid
from reporting.replay import ReplayConfig, replay

from .conftest import vd_from

CFG0 = ReplayConfig(apy=0.0, expiry_cap=None)


def one(vd, alloc, cfg=CFG0, mask=None):
    return replay(vd, np.ones(vd.n, bool) if mask is None else mask, alloc, cfg)


def test_sizing_floor():
    vd = vd_from([{"entry": "2020-01-06", "exit": "2020-02-07", "coll": 300, "pnl": 10}])
    r = one(vd, 0.10)
    assert r.contracts[0] == 33                      # floor(10_000 / 300)
    assert r.metrics["final_equity"] == pytest.approx(100000 + 330)


def test_expiry_cap_binds():
    vd = vd_from([{"entry": "2020-01-06", "exit": "2020-02-07", "coll": 1000, "pnl": 1}])
    assert one(vd, 1.0, ReplayConfig(apy=0, expiry_cap=0.60)).contracts[0] == 60
    assert one(vd, 1.0, CFG0).contracts[0] == 100


def test_unreserved_collateral_limits_second_entry():
    rows = [{"entry": "2020-01-06", "exit": "2020-02-07", "coll": 1000, "pnl": 0, "expiry": "2020-02-07"},
            {"entry": "2020-01-13", "exit": "2020-02-14", "coll": 1000, "pnl": 0, "expiry": "2020-02-14"}]
    r = one(vd_from(rows), 0.5, ReplayConfig(apy=0, expiry_cap=0.6))
    assert list(r.contracts) == [30, 20]             # 60% cap on first expiry, remaining 40% budget on the next


def test_cannot_afford_is_skipped_and_counted():
    vd = vd_from([{"entry": "2020-01-06", "exit": "2020-02-07", "coll": 5000, "pnl": 100}])
    r = one(vd, 0.01)                                 # $1000 budget < $5000 collateral
    assert r.contracts[0] == 0 and r.metrics["skipped_cant_afford"] == 1 and r.metrics["n_trades"] == 0


def test_equal_split_across_tickers():
    rows = [{"entry": "2020-01-06", "exit": "2020-02-07", "coll": 400, "pnl": 1, "ticker": "SPY"},
            {"entry": "2020-01-06", "exit": "2020-02-07", "coll": 400, "pnl": 1, "ticker": "QQQ"}]
    r = one(vd_from(rows), 0.20)
    assert list(r.contracts) == [25, 25]              # 10% each of 100k / 400
    r1 = one(vd_from(rows), 0.20, mask=np.array([True, False]))
    assert list(r1.contracts) == [50, 0]              # single ticker gets the whole 20%


def test_broke_detection_and_floor_at_zero():
    vd = vd_from([{"entry": "2020-01-06", "exit": "2020-02-07", "coll": 1000, "pnl": -1500, "reason": "stop"},
                  {"entry": "2020-03-02", "exit": "2020-04-03", "coll": 1000, "pnl": 100}],
                 {"T000": {"2020-01-06": 0, "2020-01-20": -1400}})
    r = one(vd, 1.0)
    assert r.metrics["broke"] and r.metrics["broke_date"] != ""
    assert r.equity.min() >= 0 and r.metrics["final_equity"] == 0
    assert r.contracts[1] == 0                         # no trading after broke


def test_not_broke_when_equity_above_cheapest_entry():
    vd = vd_from([{"entry": "2020-01-06", "exit": "2020-02-07", "coll": 1000, "pnl": -1000}])
    r = one(vd, 0.5)
    assert not r.metrics["broke"] and r.metrics["final_equity"] == pytest.approx(50000)


def test_marks_drive_drawdown_before_exit():
    vd = vd_from([{"entry": "2020-01-06", "exit": "2020-02-07", "coll": 1000, "pnl": 50}],
                 {"T000": {"2020-01-06": 0, "2020-01-20": -800, "2020-02-03": -200}})
    r = one(vd, 0.5)
    assert r.metrics["max_dd"] == pytest.approx(0.40)       # 50 contracts x -$800 mark on 2020-01-20, still open
    assert r.metrics["final_equity"] == pytest.approx(100000 + 50 * 50)


def test_interest_accrues_on_idle_cash():
    vd = vd_from([{"entry": "2020-01-02", "exit": "2021-01-04", "coll": 9e9, "pnl": 1}])   # never affordable
    r = one(vd, 0.5, ReplayConfig(apy=0.036, expiry_cap=None))
    days = (vd.days[-1] - vd.days[0]).days
    assert r.metrics["final_equity"] == pytest.approx(100000 * 1.036 ** (days / 365), rel=1e-6)


def test_interest_only_on_unreserved_cash():
    vd = vd_from([{"entry": "2020-01-02", "exit": "2021-01-04", "coll": 1000, "pnl": 0}])
    r = replay(vd, np.ones(1, bool), 1.0, ReplayConfig(apy=0.036, expiry_cap=None))   # whole account reserved
    assert r.metrics["final_equity"] == pytest.approx(100000, rel=1e-9)


def test_paired_replays_are_deterministic_and_share_dates():
    rows = [{"entry": f"2020-0{m}-03", "exit": f"2020-0{m}-28", "coll": 500, "pnl": p, "expiry": f"2020-0{m}-28"} for m, p in zip(range(1, 7), [10, -50, 20, 30, -10, 5])]
    vd = vd_from(rows)
    a1, a2 = one(vd, 0.15), one(vd, 0.15)
    assert np.array_equal(a1.equity, a2.equity)
    b = one(vd, 0.50)
    assert (b.days == a1.days).all() and len(b.equity) == len(a1.equity)


def test_circuit_breaker_pauses_after_losing_month():
    rows = [{"entry": "2020-01-06", "exit": "2020-01-28", "coll": 1000, "pnl": -500, "expiry": "2020-01-28"},
            {"entry": "2020-02-03", "exit": "2020-02-20", "coll": 1000, "pnl": 100, "expiry": "2020-02-20"},
            {"entry": "2020-04-01", "exit": "2020-04-20", "coll": 1000, "pnl": 100, "expiry": "2020-04-20"}]
    vd = vd_from(rows)
    off = one(vd, 0.1)
    on = replay(vd, np.ones(3, bool), 0.1, ReplayConfig(apy=0, expiry_cap=None, cb_days=21))
    assert off.metrics["n_trades"] == 3
    assert on.metrics["skipped_cb"] >= 1 and on.metrics["n_trades"] < 3


def test_too_few_trades_flag():
    vd = vd_from([{"entry": "2020-01-06", "exit": "2020-02-07", "coll": 300, "pnl": 10}])
    assert one(vd, 0.1).metrics["too_few_trades"]


def test_scenario_id_and_subsets():
    assert len(list(grid.subsets(["A", "B", "C", "D", "E"]))) == 31
