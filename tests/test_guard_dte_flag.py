"""guard_matched --dte-target: 45 by default (output unchanged), 35 selects 35-DTE entries and records it, and only the
values the run and download CLIs accept are allowed."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from playground.guard_matched import DTE_CHOICES, build_parser, main, unguarded_config  # noqa: E402

SYNTH = ["--provider", "synthetic", "--syn-start", "2015-01-02", "--syn-end", "2016-12-30",
         "--start", "2016-01-04", "--end", "2016-06-30"]


def test_allowed_values_match_the_run_and_download_cli():
    from playground.cli_lab import add_lab_flags
    import argparse
    p = argparse.ArgumentParser()
    add_lab_flags(p)
    lab = next(a for a in p._actions if a.dest == "dte_target")
    assert tuple(lab.choices) == DTE_CHOICES == (30, 35, 45, 60, 90)


def test_default_is_45_dte_and_the_arm_configs_keep_it():
    assert build_parser().parse_args([]).dte_target == 45
    assert unguarded_config("monthly", "2016-01-04", "2016-06-30").dte_target == 45
    assert unguarded_config("weekly", "2016-01-04", "2016-06-30", 45).dte_target == 45


@pytest.mark.parametrize("value", DTE_CHOICES)
def test_each_allowed_value_is_accepted(value):
    assert build_parser().parse_args(["--dte-target", str(value)]).dte_target == value


@pytest.mark.parametrize("value", [0, 28, 31, 40, 50, 120, -35])
def test_values_outside_the_allowed_set_are_rejected(value, capsys):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--dte-target", str(value)])
    assert e.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_dte_35_selects_35_dte_entries_and_records_it_in_the_metadata(tmp_path):
    out = tmp_path / "out35"
    assert main(SYNTH + ["--dte-target", "35", "--cadence", "weekly", "monthly", "--out", str(out)]) == 0
    for cadence in ("weekly", "monthly"):
        meta = json.loads((out / cadence / "summary.json").read_text())
        assert meta["switches"]["unguarded"]["dte_target"] == 35
        assert meta["switches"]["guarded"]["dte_target"] == 35
        trades = pd.read_csv(out / cadence / "trades_unguarded.csv")
        assert len(trades) > 0
        assert trades.dte_entry.between(28, 42).all()                    # the 35-DTE window: [28, 35, 42]
    monthly = pd.read_csv(out / "monthly" / "trades_unguarded.csv")
    assert set(monthly.dte_entry) == {35}                                # monthly entries are cut at exactly 35 DTE


def test_default_run_records_45_dte_in_the_metadata(tmp_path):
    out = tmp_path / "out45"
    assert main(SYNTH + ["--cadence", "monthly", "--out", str(out)]) == 0
    meta = json.loads((out / "monthly" / "summary.json").read_text())
    assert meta["switches"]["unguarded"]["dte_target"] == 45
    trades = pd.read_csv(out / "monthly" / "trades_unguarded.csv")
    assert trades.dte_entry.between(38, 52).all()
