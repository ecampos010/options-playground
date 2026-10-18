"""Biweekly entry cadence: 14-day schedule on monthly expiries nearest 35 DTE; the other cadences are unchanged."""
import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from playground.calendar import trading_days  # noqa: E402
from playground.cli import build_parser  # noqa: E402
from playground.cli_lab import add_lab_flags  # noqa: E402
from playground.config import build_config  # noqa: E402
from playground.lab import biweekly_schedule, entry_schedule, monthly_expiries, third_friday  # noqa: E402

D = dt.date
DTE_35 = [28, 35, 42]                                   # cfg.dte for dte_target=35 (the lab's +/-7 expansion)


def weekdays(start, end):
    """Every Monday-to-Friday date: no holidays, so the 14-day grid is never shifted."""
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def bio_cfg():
    return build_config("green", {"entry_cadence": "biweekly", "dte_target": 35}).replace(dte=DTE_35)


def window(days, start, end):
    i0 = next(i for i, x in enumerate(days) if x >= start)
    i1 = max(i for i, x in enumerate(days) if x <= end)
    return i0, i1


# ---------------------------------------------------------------- schedule: 14-day grid and the monthly snapping rule
def test_entries_fall_14_calendar_days_apart_from_the_window_start():
    days = weekdays(D(2016, 1, 4), D(2017, 12, 29))
    i0, i1 = window(days, D(2016, 1, 4), D(2017, 12, 29))
    sched = biweekly_schedule(days, i0, i1, bio_cfg())
    start = days[i0]
    assert sched, "the window must produce entries"
    assert all((e - start).days % 14 == 0 for e in sched)                      # every entry sits on the 14-day grid
    entries = sorted(sched)
    assert all((b - a).days % 14 == 0 for a, b in zip(entries, entries[1:]))  # skips leave gaps of whole periods


def test_every_grid_date_with_a_monthly_expiry_in_range_is_an_entry():
    days = weekdays(D(2016, 1, 4), D(2017, 12, 29))
    i0, i1 = window(days, D(2016, 1, 4), D(2017, 12, 29))
    sched = biweekly_schedule(days, i0, i1, bio_cfg())
    expiries = monthly_expiries(days, i0, i1, bio_cfg())
    expected = set()
    for k in range(0, 200):
        e = days[i0] + dt.timedelta(days=14 * k)
        if e > days[i1]:
            break
        if any(28 <= (x - e).days <= 42 for x in expiries):
            expected.add(e)
    assert set(sched) == expected


def test_a_target_on_a_non_trading_day_moves_to_the_nearest_trading_day_within_4_days():
    start = D(2016, 1, 4)
    days = [x for x in weekdays(start, D(2017, 12, 29)) if x != D(2016, 2, 15)]       # 2016-02-15 is a holiday here
    i0, i1 = window(days, start, D(2017, 12, 29))
    sched = biweekly_schedule(days, i0, i1, bio_cfg())
    assert D(2016, 2, 15) not in sched
    assert D(2016, 2, 16) in sched and sched[D(2016, 2, 16)] == D(2016, 3, 18)     # Feb 16 -> Mar 18 expiry (31 DTE)


def test_a_target_with_no_trading_day_within_4_days_is_skipped():
    start = D(2016, 1, 4)
    gap = {D(2016, 2, d) for d in (10, 11, 12, 15, 16, 17, 18, 19)}                  # 2016-02-15 target: nearest is 6+ days away
    days = [x for x in weekdays(start, D(2017, 12, 29)) if x not in gap]
    i0, i1 = window(days, start, D(2017, 12, 29))
    sched = biweekly_schedule(days, i0, i1, bio_cfg())
    assert not any(e in sched for e in gap)
    assert D(2016, 3, 14) in sched                                                 # the grid carries on after the gap


# ---------------------------------------------------------------- expiry choice (real NYSE calendar)
@pytest.fixture(scope="module")
def real_window():
    days = list(trading_days(D(2015, 11, 1), D(2026, 12, 31)))
    i0, i1 = window(days, D(2016, 10, 1), D(2026, 9, 30))
    cfg = bio_cfg()
    sched = biweekly_schedule(days, i0, i1, cfg)
    return days, i0, i1, cfg, sched


def test_every_chosen_expiry_is_a_third_friday_or_its_holiday_predecessor(real_window):
    days, _, _, _, sched = real_window
    dayset = set(days)
    assert sched
    for e, x in sched.items():
        M = third_friday(x.year, x.month)
        if x == M:
            assert x.weekday() == 4
        else:                                                    # holiday Friday: the existing rule expires Thursday
            assert x.weekday() == 3 and M not in dayset and x == max(d for d in days if d < M)


def test_no_weekly_expiry_is_ever_chosen(real_window):
    _, _, _, _, sched = real_window
    for x in sched.values():
        assert x.weekday() == 3 or x.weekday() == 4
        assert 15 <= x.day <= 21 or (x.weekday() == 3 and 14 <= x.day <= 20)  # third-Friday week, or Thursday before it


def test_each_chosen_expiry_is_the_monthly_expiry_nearest_35_dte(real_window):
    days, i0, i1, cfg, sched = real_window
    expiries = monthly_expiries(days, i0, i1, cfg)
    for e, x in sched.items():
        in_range = [y for y in expiries if 28 <= (y - e).days <= 42]
        best = min(in_range, key=lambda y: (abs((y - e).days - 35), y))
        assert x == best, (e, x, best)


def test_consecutive_entries_share_an_expiry_only_at_the_edge_of_the_range(real_window):
    _, _, _, _, sched = real_window
    entries = sorted(sched)
    for a, b in zip(entries, entries[1:]):
        if sched[a] == sched[b]:
            assert (sched[a] - a).days == 42 or (sched[a] - b).days == 28


# ---------------------------------------------------------------- the other cadences are unchanged from main
GOLDEN = {                                    # sha256 of the sorted schedule, recorded on origin/main (2016-2017)
    "monthly": (24, "ee9e62b17a1ad7600bd2d26f56a6fa72516f66dca10f8be097e1c8d408f92695"),
    "monthly_scalein4": (96, "7d33e6922d548a559bb04d098c2f2f0f59734fcb96c1a277c64a46d5c1ada1f2"),
    "weekly": (104, "3032af3bedf02ce05aa782e094273acaaf88c39a07c279e3db8873dbfc6f5560"),
    "twice-weekly": (208, "f6b8038438e73a5d552e02d410bdd4a9a663ce73b62643ad0b4ded6a162eca18"),
    "daily": (503, "ded1206cc1b397c44dcd3efec0077c6ca56a775590ed9df2d23871755f185b4f"),
}


@pytest.mark.parametrize("name,over", [
    ("monthly", {"entry_cadence": "monthly"}),
    ("monthly_scalein4", {"entry_cadence": "monthly", "scale_in": 4}),
    ("weekly", {"entry_cadence": "weekly"}),
    ("twice-weekly", {"entry_cadence": "twice-weekly"}),
    ("daily", {"entry_cadence": "daily"}),
])
def test_existing_cadences_produce_the_same_schedule_as_main(name, over):
    days = list(trading_days(D(2015, 11, 1), D(2018, 12, 31)))
    i0, i1 = window(days, D(2016, 1, 4), D(2017, 12, 29))
    cfg = build_config("green", {**over, "dte_target": 35}).replace(dte=DTE_35)
    sched, _ = entry_schedule(days, i0, i1, cfg)
    blob = repr(sorted((str(k), str(v)) for k, v in sched.items()))
    assert (len(sched), hashlib.sha256(blob.encode()).hexdigest()) == GOLDEN[name]


def test_entry_schedule_routes_biweekly_to_the_new_schedule_only():
    days = list(trading_days(D(2015, 11, 1), D(2017, 12, 31)))
    i0, i1 = window(days, D(2016, 1, 4), D(2017, 12, 29))
    sched, _ = entry_schedule(days, i0, i1, bio_cfg())
    assert sched == biweekly_schedule(days, i0, i1, bio_cfg())


# ---------------------------------------------------------------- CLI
def _lab_parser():
    p = argparse.ArgumentParser()
    add_lab_flags(p)
    return p


def test_entry_cadence_choices_are_the_four_existing_ones_plus_biweekly():
    action = next(a for a in _lab_parser()._actions if a.dest == "entry_cadence")
    assert action.choices == ["monthly", "weekly", "twice-weekly", "daily", "biweekly"]


@pytest.mark.parametrize("value", ["monthly", "weekly", "twice-weekly", "daily", "biweekly"])
def test_each_accepted_cadence_parses(value):
    assert _lab_parser().parse_args(["--entry-cadence", value]).entry_cadence == value


def test_an_unknown_cadence_is_still_rejected():
    with pytest.raises(SystemExit):
        _lab_parser().parse_args(["--entry-cadence", "fortnightly"])


@pytest.mark.parametrize("argv", [
    ["run", "--ticker", "SPY", "--entry-cadence", "biweekly"],
    ["download", "--engine", "lab", "--tickers", "SPY", "--entry-cadence", "biweekly"],
])
def test_run_and_download_accept_biweekly(argv):
    args = build_parser().parse_args(argv)
    assert args.entry_cadence == "biweekly"


def test_plan_accepts_biweekly_in_its_cadence_list():
    args = build_parser().parse_args(["plan", "--engine", "lab", "--entry-cadences", "monthly,biweekly"])
    assert args.entry_cadences == "monthly,biweekly"
