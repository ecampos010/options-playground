# options-playground architecture (origin/main at 4c5a493, PR #16)
Read from a fresh `git fetch` of main; nothing run except `pytest --collect-only`. Research software, not advice.

## 1. Layout
- `playground/` engine and data layer. `lab.py` (lab engine), `engine.py` (older single-ticker Green engine used by
  `backtest`/`matrix`/`ladder`), `rules.py` (guard/exit/sizing functions), `strategies.py`, `mgmt.py` (triggers, rolls,
  hedges), `config.py`, `schema.py` (trade-log contract), `guard_matched.py`, `sweep.py`, `stress.py`.
  Data: `provider.py` (interface + SyntheticProvider), `theta_provider.py` + `cache.py` (parquet cache), `theta_client.py`
  (REST adapter), `downloader.py`, `spot.py` (Alpaca/Massive/parity spot), `exdiv.py`, `dailydata.py`, `errorlog.py`.
  Misc: `cli.py` + `cli_lab.py` (entry points), `planner.py`, `probe.py`, `housekeeping.py` (export/purge), `reserve.py` (stubs).
- `reporting/` separate package; reads trade logs only: `schema.py`, `replay.py`, `grid.py`, `montecarlo.py`, `thermo.py`,
  `playbooks.py`, `why.py`, `stress.py`, `pipeline.py`, `htmlreport.py` + `assets/`, `synth.py` (fake logs), `reporting/tests/`.
- `profiles/*.yaml` (green, yellow_E1..E7, red, experiment_*, `tickers.yaml`); `config/*.yaml` (reporting thresholds);
  `sweeps/*.yaml`; `scripts/` (`fetch_context_data.py` -> data/events.csv + rates.csv; `staggering_study.py`);
  `docs/` (TRADE_LOG_SCHEMA, RUN_ORDER, GUARD_MATCHED_REPORT, SCHEMA_REQUESTS); `tests/`; `data/stress_windows.csv` (only tracked file in data/).

## 2. Pipeline
ThetaData terminal -> `download` -> `data/` parquet cache -> `ThetaProvider` -> lab engine -> `runs/<run_id>/` -> sweep summary / `reporting` -> `index.html`.
- Download: `python -m playground download --engine lab --tickers SPY ...` (Stage A entry snapshot, Stage B legs on demand,
  resumable, <=4 concurrent requests, failures to `errors.csv`). Spot bars: `python -m playground spot`. Context: `scripts/fetch_context_data.py`.
- Lab run: `python -m playground run ...` (one run). Sweep: `python -m playground sweep sweeps/spy_core.yaml`
  (resumable by config hash; prints configs tried and first/second-half return on risk). Stress: `... stress`.
- Guard comparison: `python -m playground.guard_matched --provider theta --data-dir data --events data/events.csv`
  (own `__main__`; not a subcommand of `playground`).
- Report: `python -m reporting validate runs/`, `python -m reporting report --runs runs/ --out report/`; synthetic: `python -m reporting demo`.
- Each run dir holds trades.csv, marks.csv, blocked_entries.csv, config.json, meta.json (see section 3).

## 3. Lab engine (`playground/lab.py`, ~1050 lines, class `_Lab`; entry `run_lab`)
- Entry schedule: `entry_schedule()` returns date -> expiry hint. Cadences monthly, weekly, twice-weekly, daily, biweekly
  (monthly expiries only); scale-in 4. `dte_target` becomes the window [t-7, t, t+7]. `entry_days=` restricts entries to a given set.
- Strategies (`strategies.py`): iron_condor, iron_butterfly, bull_put_spread, put_bwb, cash_secured_put, covered_call, long_put,
  put_debit_spread, long_strangle. Strikes by VIX distance or delta (Greeks, else Black-Scholes from mid IV). Max loss from exact payoff.
- Entry guards from `rules.py`: gate, upside, event buffer, circuit breaker (equity mode only), term, spike, IV rank, delta cap.
- Exits: stop (natural close cost vs credit), take-profit, breach, time exit, forced close (fly), dollar stop, triggers
  (distance/delta) with actions (exit, rolls, close tested spread, hedges). Detected on one bar, filled on the NEXT bar at natural prices.
- Sizing: `equity-pct` (budget % of current equity, 60% per-expiry cap, floor/round contracts) or `fixed1` (unit mode: 1 contract, no budget, no breaker).
- Quote checks: `valid_quote`, `quote_sanity_faults` (exit triggers skipped on bars with bad quotes), min credit, missing-spot days blocked
  as `nodata`, counterfactual `cf_pnl_to_expiry` from expiry payoff; positions open at data end go to `meta.open_excluded`, not trades.csv.
- Schema (`schema.py`, validator `validate_run_dir`): 32 trade columns, per-day marks, blocked reasons, config/meta. Money is USD per one contract.

## 4. Guard harness (`playground/guard_matched.py`)
- `_arm()` builds both arms from a fresh `Config()`: SPY iron condor, delta 0.16 both sides, dte_target 45, 5m, tp 0.5, stop 2.0,
  breach on, time exit 21, natural fills, equity-pct, spot alpaca with parity fallback. Not read from profiles/*.yaml.
- Green-Unguarded: gate, upside_guard, event_buffer, earnings_filter off and cb_days 0. Guarded: those five at Green defaults. Nothing else differs.
- Schedule: `lab.entry_schedule` on the cache's SPY daily index, same dte_target expansion as the engine (`_scheduled_days`).
  The unguarded arm runs first; the guarded arm runs with `entry_days=` the unguarded entry dates (open-at-end included), so guarded entries are a subset.
- CLI: `--cadence monthly weekly [biweekly]`, `--dte-target 30|35|45|60|90`, `--start/--end`. Output: per-arm metrics, skipped-trade table, by-year, split sample.

## 5. Configs
- `Config` dataclass (`config.py`): equity 100000, apy 0.036, budget None -> tier (green 15%, yellow 5%, red 1%, experiment 5%),
  expiry cap 0.60, gate VIX>=25 / day<=-1.5% / 5d<=-4%, upside +4%, event buffer 2 days, cb 21, dte [30,35,45], vix-distance strikes
  (4.6% puts, 4.0% calls) or delta 0.16, tp 0.50, stop 2.0, time exit 21, fill natural, entry 09:45-15:30 random seed 1, res 5m.
  Lab: ticker SPY, iron_condor, equity-pct, weekly, width 0.7% of spot (SPY fixed $5 in tickers.yaml), min credit 10% of width, fee 0.05/leg, spot alpaca.
- Profiles are flat `key: JSON` files that override Config keys (own parser, no PyYAML). `level` 1 green, 2 yellow (each differs from green by one
  switch, tested), 3 red (enter only when the gate trips), 4 experiment (guards off). `META_KEYS` (profile, tier, budget_pct, level) are not counted as switches.

## 6. Tests
- `tests/` (engine, lab, theta plumbing, spot, sweep/stress, guard, schema fixture) and `reporting/tests/` (replay, grid, report, why, playbooks).
  Run `python -m pytest tests -q` and `python -m pytest reporting/tests -q`. Collected on main: 300 + 85 = 385 (not executed here).
  Offline: SyntheticProvider and `fake_theta.py` stand in for ThetaData. Two reporting browser tests skip without Playwright/Chromium.

## 7. HTML report
- Generator: `reporting/htmlreport.py` (`build_html`), called by `reporting/pipeline.py` `run_report`. Inlines `reporting/assets/report.css` and `report.js`.
- Reads: `runs/*/{trades,marks,blocked_entries}.csv`, `config.json`, `meta.json` (via `reporting/schema.load_runs`); `config/{thermometer,verdicts,
  ways_to_lose,playbooks,why}.yaml`; optional `data/rates.csv`, `data/events.csv`, `data/underlying/{SPY,VIX}_daily.parquet`, `--spy-csv`.
- Writes `<out>/index.html` (one file) plus results.csv, verdicts.md, ways_to_lose_money.md, stress.md, playbooks.md, why.md, equity/*.parquet.
  Synthetic: `python -m reporting demo` -> `runs_synth/` then `report_synth/index.html` (both gitignored).
- A committed synthetic page is NOT on main. `docs/demo_report/index.html` (~5 MB, SYNTHETIC banner) exists only on branch `claude/dazzling-meitner-syu8xx`.
- BWB: no dedicated section in htmlreport/pipeline/config. BWB appears only as rows in the generic tables, from synthetic strategy
  `put_broken_wing_butterfly` (`synth.py`). Nothing in `reporting/` references the lab strategy name `put_bwb`.

## 8. Known gaps
- `reserve.py`: `overlay` and `spike-study` are stubs (CLI says "RESERVED - not built", raises NotImplementedError).
- TODO-VERIFY / unverified: bulk `strike=*`, option EOD, parameter formats in `theta_client.py`; Alpaca and Massive spot paths in `spot.py`;
  Greeks endpoint fields; per-ticker notes in `tickers.yaml`; Cboe vol-index CSV URLs.
- `GUARD_MATCHED_REPORT.md`: code and tests done, no real-data results yet. Provider base methods raise NotImplementedError (abstract).
- Reporting limits it states itself: 10-year sample, bootstrap understates tails, linear sizing, no taxes or margin interest, no assignment model beyond `exdiv_flag`.
