"""guard_matched scheduled-day count: the same schedule the lab run uses, so count and dates agree."""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from playground.calendar import trading_days  # noqa: E402
from playground.config import build_config  # noqa: E402
from playground.guard_matched import _scheduled_days, unguarded_config  # noqa: E402
from playground.lab import entry_schedule  # noqa: E402

D = dt.date
DAYS = list(trading_days(D(2015, 11, 1), D(2026, 12, 31)))


class _Provider:
    def get_underlying_daily(self):
        return pd.DataFrame({"close": [1.0] * len(DAYS)}, index=DAYS)


def _window(start, end):
    i0 = max(6, next(i for i, x in enumerate(DAYS) if str(x) >= start))
    i1 = max(i for i, x in enumerate(DAYS) if str(x) <= end)
    return i0, i1


def _lab_dates(cfg):
    """The dates the lab run schedules: its own expansion of dte_target, then entry_schedule."""
    t = cfg.dte_target
    lab_cfg = cfg.replace(dte=[t - 7, t, t + 7]) if t else cfg
    i0, i1 = _window(cfg.start_date, cfg.end_date)
    return set(entry_schedule(DAYS, i0, i1, lab_cfg)[0])


@pytest.mark.parametrize("cadence", ["monthly", "biweekly", "weekly"])
@pytest.mark.parametrize("dte", [35, 45])
def test_count_equals_the_lab_schedule_for_35_and_45_dte(cadence, dte):
    cfg = unguarded_config(cadence, "2016-10-01", "2026-09-30", dte)
    assert _scheduled_days(_Provider(), cfg) == len(_lab_dates(cfg))


@pytest.mark.parametrize("dte", [35, 45])
def test_monthly_count_matches_the_lab_dates_with_the_offset_applied(dte):
    cfg = unguarded_config("monthly", "2016-10-01", "2026-09-30", dte)
    assert len(_lab_dates(cfg)) == 120
    assert _scheduled_days(_Provider(), cfg) == 120


def test_a_45_dte_monthly_entry_is_scheduled_at_a_45_day_offset_not_35():
    cfg = unguarded_config("monthly", "2016-10-01", "2026-09-30", 45)
    t = cfg.dte_target
    assert t == 45
    wrong = set(entry_schedule(DAYS, *_window("2016-10-01", "2026-09-30"), cfg.replace(dte=[30, 35, 45]))[0])
    assert _lab_dates(cfg) != wrong          # the unexpanded window (the old count) picks different dates


def test_without_a_dte_target_the_count_is_the_config_as_given():
    cfg = build_config("green", {"entry_cadence": "monthly", "start_date": "2016-10-01", "end_date": "2026-09-30"})
    assert cfg.dte_target is None
    i0, i1 = _window("2016-10-01", "2026-09-30")
    assert _scheduled_days(_Provider(), cfg) == len(entry_schedule(DAYS, i0, i1, cfg)[0])


def test_documented_counts_on_the_default_window():
    assert _scheduled_days(_Provider(), unguarded_config("monthly", "2016-10-01", "2026-09-30", 45)) == 120
    assert _scheduled_days(_Provider(), unguarded_config("weekly", "2016-10-01", "2026-09-30", 45)) == 522
    assert _scheduled_days(_Provider(), unguarded_config("biweekly", "2016-10-01", "2026-09-30", 35)) == 120
    assert _scheduled_days(_Provider(), unguarded_config("biweekly", "2016-10-01", "2026-09-30", 45)) == 128
