import pytest

from playground.cli import build_parser, main, overrides_from
from playground.config import build_config

SYN = ["--provider", "synthetic", "--syn-start", "2016-01-04", "--syn-end", "2016-12-30", "--resolution", "15m"]


def test_matrix_cli(capsys, tmp_path):
    assert main(["matrix", *SYN, "--results-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "MATRIX" in out and out.count("tp") >= 8 and "fill" in out


def test_matrix_fill_compare_cli(capsys, tmp_path):
    assert main(["matrix", *SYN, "--fill-compare"]) == 0
    out = capsys.readouterr().out
    assert "final[natural]" in out and "final[mid]" in out


def test_ladder_cli(capsys):
    assert main(["ladder", *SYN, "--equal-budget"]) == 0
    out = capsys.readouterr().out
    assert "yellow_E7" in out and "red" in out and "fill=natural" in out
    assert "TOO FEW TRADES - not evidence" in out or "10-year tier only" in out


def test_backtest_cli_compact_and_files(capsys, tmp_path):
    assert main(["backtest", *SYN, "--results-dir", str(tmp_path), "--run-id", "r1", "--fill", "haircut",
                 "--entry-sample", "every-30m", "--cb", "0", "--tp", "0.75", "--time-exit", "0"]) == 0
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) <= 40 and "fill=haircut(0.05)" in out
    for f in ("trades.csv", "equity.csv", "config.json", "summary.md"):
        assert (tmp_path / "r1" / f).exists()


def test_missing_events_is_clean_error(capsys, tmp_path):
    rc = main(["backtest", "--provider", "cache", "--data-dir", str(tmp_path), "--events", str(tmp_path / "x.csv")])
    assert rc == 2 and "never invented" in capsys.readouterr().out


def test_reserved_commands(capsys):
    assert main(["overlay"]) == 3 and main(["spike-study"]) == 3


# ---------------------------------------------------------------- --dte-target choices
@pytest.mark.parametrize("cmd", [["run"], ["plan"], ["stress"], ["download"]])
@pytest.mark.parametrize("v", [30, 35, 45, 60, 90])
def test_dte_target_accepts_supported_values(cmd, v):
    args = build_parser().parse_args([*cmd, "--dte-target", str(v)])
    assert args.dte_target == v
    assert build_config("green", overrides_from(args)).dte_target == v


@pytest.mark.parametrize("v", ["40", "0", "120", "-30"])
def test_dte_target_rejects_unlisted_values(v, capsys):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["run", "--dte-target", v])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "--dte-target" in err and "choose from '30', '35', '45', '60', '90'" in err


@pytest.mark.parametrize("v", ["35.5", "thirty"])
def test_dte_target_rejects_non_integers(v, capsys):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["run", "--dte-target", v])
    assert e.value.code == 2 and "--dte-target" in capsys.readouterr().err
