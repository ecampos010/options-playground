"""guard_matched --cadence biweekly: accepted, and scheduled by the lab engine's biweekly schedule."""
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from playground.calendar import trading_days  # noqa: E402
from playground.guard_matched import build_parser, unguarded_config  # noqa: E402
from playground.lab import biweekly_schedule, entry_schedule  # noqa: E402

D = dt.date


def test_cadence_accepts_biweekly_alongside_monthly_and_weekly():
    ap = build_parser()
    cad = next(a for a in ap._actions if a.dest == "cadence")
    assert list(cad.choices) == ["monthly", "weekly", "biweekly"]
    assert ap.parse_args(["--cadence", "biweekly"]).cadence == ["biweekly"]
    assert ap.parse_args(["--cadence", "monthly", "weekly"]).cadence == ["monthly", "weekly"]


def test_harness_biweekly_schedule_is_the_lab_engine_schedule():
    days = list(trading_days(D(2015, 11, 1), D(2026, 12, 31)))
    i0 = next(i for i, x in enumerate(days) if str(x) >= "2016-10-01")
    i1 = max(i for i, x in enumerate(days) if str(x) <= "2026-09-30")
    cfg = unguarded_config("biweekly", "2016-10-01", "2026-09-30", 35)
    assert cfg.entry_cadence == "biweekly"
    # the lab engine expands dte_target 35 to [28, 35, 42] before it schedules (lab.py _Lab.__init__)
    lab_cfg = cfg.replace(dte=[28, 35, 42])
    lab_sched, _ = entry_schedule(days, i0, i1, lab_cfg)
    assert lab_sched == biweekly_schedule(days, i0, i1, lab_cfg)
    assert len(lab_sched) > 100
