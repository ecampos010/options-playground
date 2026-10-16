"""Tests for scripts/staggering_study.py: schedule sizing, blocked tranches, and regression against the guard study."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(path, name, strip_main=False):
    src = Path(path).read_text()
    if strip_main:  # the guard study calls main() at import time
        lines = src.rstrip().splitlines()
        assert lines[-1].strip() == "main()"
        src = "\n".join(lines[:-1])
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def st():
    spec = importlib.util.spec_from_file_location("staggering_study", ROOT / "scripts" / "staggering_study.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


@pytest.fixture(scope="module")
def guard():
    return _load(ROOT / "claude" / "guard-calibration-study.py.txt", "guard", strip_main=True)


@pytest.fixture(scope="module")
def data(st):
    px, vix = st.synthetic(n=1200, seed=3)
    return st.prepare(px, vix, "1900-01-01")


# ---------------------------------------------------------------- sizing
@pytest.mark.parametrize("name", ["A_ALL_AT_ONCE", "B_TWO_TRANCHES", "C_FOUR_TRANCHES", "D_TWO_TRANCHES_WIDE"])
def test_schedule_fractions_and_max_loss_sum_to_budget(st, name):
    sched = st.SCHEDULES[name]
    assert sum(f for _, f in sched) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        width = rng.uniform(2, 20); credits = rng.uniform(0.2, 0.45, len(sched)) * width  # each tranche has its own credit
        t = st.tranche_sizing(15000.0, sched, credits, width)
        assert sum(x["max_loss"] for x in t) == pytest.approx(15000.0)
        for x, (_, f) in zip(t, sched): assert x["max_loss"] == pytest.approx(f * 15000.0)


def test_tranche_pnl_matches_fraction_times_ror(st):
    width, cr, cost = 5.0, 1.5, 0.75
    (t,) = st.tranche_sizing(15000.0, [(0, 0.5)], [cr], width)
    pnl = t["contracts"] * (cr - cost) * 100
    assert pnl == pytest.approx(0.5 * 15000.0 * (cr - cost) / (width - cr))


# ---------------------------------------------------------------- blocked tranches
def _fake(n=40):
    sim = pd.DataFrame({"ror": np.linspace(-1, 1, n) * 0.5, "why": ["tp"] * n, "cr": 0.3, "k": 5})
    return sim


def test_blocked_tranche_is_skipped_not_delayed_and_counted(st):
    sim = _fake(); n = len(sim); blocked = np.zeros(n, bool); blocked[13] = True  # block only the +3 tranche of start 10
    starts = np.array([10, 20])
    r = st.cycle_results(sim, blocked, starts, st.SCHEDULES["B_TWO_TRANCHES"], True)
    exp0 = 0.5 * sim.ror[10] + 0.0                      # delayed tranche skipped, nothing re-entered on day 14
    exp1 = 0.5 * sim.ror[20] + 0.5 * sim.ror[23]
    assert r["ret"] == pytest.approx([exp0, exp1])
    assert r["blocked"].tolist() == [[False, True], [False, False]]
    off = st.cycle_results(sim, blocked, starts, st.SCHEDULES["B_TWO_TRANCHES"], False)      # gates OFF ignores the block
    assert off["ret"][0] == pytest.approx(0.5 * sim.ror[10] + 0.5 * sim.ror[13]) and not off["blocked"].any()


def test_fully_blocked_cycle_is_flat_and_has_no_stops(st):
    sim = _fake(); sim["why"] = "stop"; blocked = np.ones(len(sim), bool)
    r = st.cycle_results(sim, blocked, np.array([5]), st.SCHEDULES["C_FOUR_TRANCHES"], True)
    assert r["ret"][0] == 0 and r["blocked"].all() and not r["cluster"][0]


def test_same_five_day_stop_cluster(st):
    sim = _fake(); sim["why"] = "stop"; sim["k"] = 2
    blocked = np.zeros(len(sim), bool)
    # A: single tranche stopping = 100% of budget inside the window
    assert st.cycle_results(sim, blocked, np.array([5]), st.SCHEDULES["A_ALL_AT_ONCE"], True)["cluster"][0]
    # C: stops at days 2,5,9,12: any 5-day window holds at most 2 tranches = 50%, which is not "more than half"
    assert not st.cycle_results(sim, blocked, np.array([5]), st.SCHEDULES["C_FOUR_TRANCHES"], True)["cluster"][0]
    sim["k"] = 1   # stops at days 1,4,8,11 -> windows of 5 days still max 2 tranches
    assert not st.cycle_results(sim, blocked, np.array([5]), st.SCHEDULES["C_FOUR_TRANCHES"], True)["cluster"][0]
    # B with k=2: stops at +2 and +5 -> both in one 5-day window = 100%
    assert st.cycle_results(sim.assign(k=2), blocked, np.array([5]), st.SCHEDULES["B_TWO_TRANCHES"], True)["cluster"][0]


def test_gate_rule(st):
    d = pd.DataFrame({"vix": [20, 25, 20, 20, 20], "d1": [0, 0, -.015, 0, -.0149], "d5": [0, 0, 0, -.04, -.039]})
    assert st.gate_blocked(d).tolist() == [False, True, True, True, False]


# ---------------------------------------------------------------- regression vs the guard study
def test_simulate_reproduces_guard_study_at_35_dte(st, guard, data):
    mine = st.simulate(data, dte=35); theirs = guard.simulate(data)
    assert st.horizon(35) == (guard.HT, guard.TX)
    np.testing.assert_allclose(mine.ror.values, theirs.ror.values, rtol=0, atol=1e-12, equal_nan=True)
    assert (mine.why.values == theirs.why.values).all()
    np.testing.assert_allclose(mine.cr.values, theirs.cr.values, atol=1e-12)


def test_all_at_once_reproduces_guard_clean_day_numbers(st, guard, data):
    d = data; sim_g = guard.simulate(d)
    hg = (d.vix >= 25) | (d.d1 <= -.015); mg = (d.d1 >= .015) | (d.d5.abs() >= .04); clean = (~(hg | mg)).values
    ref = sim_g[clean & sim_g.ror.notna().values]
    sim = st.simulate(d, dte=35); starts = np.flatnonzero(clean & sim.ror.notna().values)
    r = st.cycle_results(sim, st.gate_blocked(d), starts, st.SCHEDULES["A_ALL_AT_ONCE"], gates_on=False)["ret"]
    assert len(r) == len(ref) > 100
    assert r.mean() == pytest.approx(ref.ror.mean(), abs=1e-12)
    assert (r > 0).mean() == pytest.approx((ref.ror > 0).mean())
    assert np.quantile(r, .05) == pytest.approx(ref.ror.quantile(.05), abs=1e-12)
    assert sim.why.iloc[starts].isin(["breach", "stop"]).mean() == pytest.approx(ref.why.isin(["breach", "stop"]).mean())


def test_schedules_equal_at_zero_dispersion(st):
    """With identical ror on every day, every schedule returns the same (sizing preserves total R)."""
    sim = pd.DataFrame({"ror": 0.2, "why": "tp", "cr": 0.3, "k": 3}, index=range(50))
    for s in st.SCHEDULES.values():
        assert st.cycle_results(sim, np.zeros(50, bool), np.array([0, 1, 2]), s, True)["ret"] == pytest.approx([0.2] * 3)


# ---------------------------------------------------------------- end to end
def test_synthetic_end_to_end_writes_outputs(st, tmp_path):
    st.main(["--synthetic", "--boot", "5", "--out", str(tmp_path)])
    for f in ("summary.md", "metrics.csv", "bootstrap_ci.csv", "crash_episodes.csv", "tranche_correlation.csv", "split_sample.csv"):
        assert (tmp_path / f).exists()
    m = pd.read_csv(tmp_path / "metrics.csv")
    assert set(m.schedule) == set(st.SCHEDULES) and set(m.variant) == {"daily", "monthly"} and set(m.gates) == {"on", "off"}
    assert (m[m.gates == "off"].tranche_blocked_pct == 0).all()
    assert "modeled, relative only" in (tmp_path / "summary.md").read_text().lower()
