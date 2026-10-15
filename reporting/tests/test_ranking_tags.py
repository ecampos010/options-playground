import json
import shutil
from pathlib import Path

import pandas as pd

from reporting import effective_level
from reporting import grid as G
from reporting.schema import load_runs, validate_cli

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "sample_run"


def test_level_tag_green_only_up_to_15pct():
    assert effective_level(1, 5) == 1 and effective_level(1, 15) == 1
    assert effective_level(1, 20) == 4 and effective_level(1, 100) == 4
    assert effective_level(2, 15) == 2 and effective_level(3, 50) == 4 and effective_level(4, 5) == 4


def test_rank_is_zone_first_then_return_over_drawdown():
    rows = [
        dict(scenario_id="breaking_hi", zone="Breaking point", calmar=9.0, max_dd=0.6, too_few_trades=False, broke=False),
        dict(scenario_id="safe_lo", zone="Safe", calmar=0.5, max_dd=0.05, too_few_trades=False, broke=False),
        dict(scenario_id="mild_hi", zone="Mild", calmar=3.0, max_dd=0.2, too_few_trades=False, broke=False),
        dict(scenario_id="safe_hi", zone="Safe", calmar=1.5, max_dd=0.1, too_few_trades=False, broke=False),
        dict(scenario_id="danger", zone="Danger", calmar=5.0, max_dd=0.4, too_few_trades=False, broke=False),
        dict(scenario_id="went_broke", zone="Breaking point", calmar=0.0, max_dd=1.0, too_few_trades=False, broke=True,
             broke_date="2020-03-16", final_equity=0.0),
    ]
    ok, br = G.rank(pd.DataFrame(rows))
    assert list(ok["scenario_id"]) == ["safe_hi", "safe_lo", "mild_hi", "danger", "breaking_hi"]
    assert list(br["scenario_id"]) == ["went_broke"]


def test_validate_accepts_single_run_folder_and_parent(tmp_path, capsys):
    assert validate_cli(str(FIX)) == 0                                  # the run folder itself
    parent = tmp_path / "runs"
    shutil.copytree(FIX, parent / "r1")
    assert validate_cli(str(parent)) == 0                               # a parent folder of runs
    out = capsys.readouterr().out
    assert out.count("delta source: bs_fallback") == 2
    lg = load_runs(FIX)
    assert lg.delta_sources == {"sample_run": "bs_fallback"}


def test_delta_source_derived_from_counts_when_tag_missing(tmp_path):
    run = tmp_path / "runs" / "r1"
    shutil.copytree(FIX, run)
    meta = json.loads((run / "meta.json").read_text())
    meta.pop("delta_source_tag")
    meta["delta_source"] = {"provider": 5, "bs_mid_iv": 0}
    (run / "meta.json").write_text(json.dumps(meta))
    assert load_runs(run.parent).delta_sources == {"r1": "theta_greeks"}
