# reporting/ - portfolio replay, ranking, thermometer, Monte Carlo ruin, stress lab

Consumes the standardized trade logs written by the backtest engine (TASK A). It never touches vendor data, option
pricing or `playground/`. Research software, not advice. Every report states:
**"10-year real-quote sample; results are relative comparisons, not forecasts."**

## Quickstart
```bash
pip install -r reporting/requirements.txt
python -m reporting demo                                  # synthetic logs -> runs_synth/, report -> report_synth/ (SYNTHETIC, ~1-2 min)
python -m reporting validate runs/                        # schema check, exit code 1 on errors
python -m reporting report --runs runs/ --out report/     # real logs (from the VPS)
python -m pytest reporting/tests -q
```
Outputs in `--out`: `index.html` (ONE self-contained file), `results.csv` (one row per scenario), `verdicts.md`,
`ways_to_lose_money.md`, `stress.md`, `equity/*.parquet` (daily equity + drawdown: top 50, bottom 20, and every broke scenario).
A compact summary (< 40 lines) is printed.

## Commands and flags
| command | purpose |
|---|---|
| `validate RUNS` | load `RUNS/*/{trades,marks,blocked_entries}.csv` + `config.json`; clear errors (missing columns, non-positive collateral, exit before entry, marks outside trade life, bad enums, unknown trade ids) |
| `make-synthetic-logs --out D [--seed 7 --start --end --cadence 5]` | deterministic SYNTHETIC fixtures: 5 tickers, 4 strategies, Levels 1-4, delta/management/wait variants, stress windows, experiments |
| `report --runs R --out O` | the full report |
| `demo` | generator + report in one go |

`report`/`demo` flags: `--allocations 5,15,20,50,100` `--apy 0.036` `--cb 0|21` (replay circuit breaker: pause entries 21 trading days after a
losing calendar month) `--expiry-cap 0.6` (0 = off) `--paths 10000` `--block 1|2|3` `--seed 7` `--no-mc` `--jobs N`
`--spy-csv PATH` (local SPY daily CSV with a date and close/adj close column; omitted line is stated in the report) `--bp-strategy iron_condor`
`--bp-tickers SPY` `--bp-profile '^green'` `--bp-dd 0.5` `--baseline-tag baseline` `--config-dir DIR`.

## What the replay does
* Start equity $100,000. Budget = allocation A x CURRENT equity, split equally across the selected tickers (and, for the
  "all_strategies_equal" mix, equally across the strategies of each ticker).
* Entry: `contracts = floor(share x equity / collateral_per_contract)`, capped by the per-expiry cap (default 60% of the ticker's share) and by unreserved collateral
  (sleeve budget and account). `contracts < 1` => skipped and counted as "could not afford".
* Realised P&L = contracts x `pnl_per_contract` at `exit_ts`; daily equity adds mark-to-market from `marks.csv`, so drawdowns include open-position losses.
* Idle (unreserved) cash earns `--apy`; collateral earns nothing (conservative).
* **Broke** = equity < collateral of the cheapest 1-contract entry in the scenario, or <= 0. First broke date is recorded, entries stop, equity is floored at 0.
* Paired: all scenarios of one profile/config share one calendar; only the knob (allocation, ticker set, cb) changes.
* Metrics: final equity, CAGR, max drawdown, worst calendar month, worst trade (USD and % of equity at entry), win rate, profit factor, return/maxDD (CAGR / max(maxDD, 0.5%)),
  average contracts, entries skipped, longest losing streak, broke flag/date, years, trades. Under 30 trades: **TOO FEW TRADES - not evidence**.

## Ranking
Scenarios that did not go broke are ranked within thermometer zone first (Safe, Mild, Danger, Breaking point), then by return/maxDD (never by final equity); within a zone, rows with < 30 trades come after the evidence-grade rows. Level tags: Green / Lvl 1 only at allocation <= 15% from Green-rule logs; larger allocations are tagged Experiment / Lvl 4.
Broke scenarios are a separate list, latest broke first. Each row has a plain-English verdict from `config/verdicts.yaml`
and a level tag ("Green / Lvl 1", "Yellow / Lvl 2", "Red / Lvl 3", "Experiment / Lvl 4"). Text carries the meaning; colour is decoration.

## Thermometer (`config/thermometer.yaml`)
```yaml
ruin_horizon_months: 36
zones:                      # first zone whose limits are STRICTLY satisfied wins
  - {name: Safe, ruin: 0.01, dd: 0.15}
  - {name: Mild, ruin: 0.05, dd: 0.30}
  - {name: Danger, ruin: 0.20, dd: 0.50}
fallback_zone: Breaking point   # anything worse, or any broke scenario
```
`ruin` is the Monte Carlo P(broke); with `--no-mc` only max drawdown and the broke flag are used (the report says so).
The VIX-regime table groups trades by `vix_bucket` and replays regime-only entries to find the highest allocation / widest delta that stayed in Safe or Mild; < 30 trades = "thin evidence".

## Monte Carlo
Block bootstrap of replayed calendar months (blocks of 1-3 months, with replacement). The same month index is used for every ticker (co-movement preserved) and the draws are shared by all scenarios (paired).
Horizons 12 and 36 months; reports P(broke), P(DD>50%), P(DD>90%), P(end < $100K), median and p10/p90 final equity, median months to ruin.
"Ways to lose money" swaps one mistake at a time into the base (Green SPY condor at 15%) using whatever logs exist; missing logs print "not run". Name matching is in `config/ways_to_lose.yaml`.

## Publishing the HTML
`report/index.html` has no external requests (inline CSS/JS/SVG, < 8 MB, light/dark via `prefers-color-scheme`, phone-safe). Publish it as a private page with
the Artifact tool / "publish as private page", or just open or email the file. For synthetic output keep the SYNTHETIC banner; do not present it as results.

## What this does NOT tell you
* **10-year sample**: one history, one set of crashes. Results are *relative* comparisons, not forecasts, and say nothing about regimes outside the sample.
* **Bootstrap limits**: resampling months cannot create a month worse than the worst one observed, so tail risk and ruin odds are understated; block length changes the answer.
* **Sizing is an assumption**: unit logs are scaled linearly by contracts; real fills at size (slippage, liquidity, margin calls, broker haircuts) are not modelled.
* **No taxes, no commissions beyond what the logs contain, no margin-interest cost.**
* **No early-assignment modelling** beyond the logged `exdiv_flag`.
* Verdicts and zones are fixed rules over those numbers, not advice.
* Synthetic fixtures prove the machinery only; their numbers mean nothing.

## Trade-log schema (contract with TASK A)
`runs/<run_id>/trades.csv` (one row per closed 1-contract position): run_id, trade_id, ticker, strategy, level (1-4), profile, config_tag, entry_ts, exit_ts (US/Eastern ISO), expiry,
dte_entry, exit_reason (tp|stop|stop_dollar|breach|trigger_exit|time|expiry|other), width, collateral_per_contract, credit_per_contract, pnl_per_contract, fees_per_contract,
adj_count, adj_cost_per_contract, hedge_cost_per_contract, cf_pnl_to_expiry_per_contract, entry_spot, exit_spot, entry_vix, vix_bucket (lt15|15to20|20to30|gt30), vix_term_ratio,
short_put_delta, short_call_delta, exdiv_flag, guards_on, fill_model, spot_source. `marks.csv`: trade_id, date, mark_pnl_per_contract. `blocked_entries.csv`: date, ticker, reason.
`config.json`, `meta.json`. All money in USD per ONE contract. Requests for changes: `docs/SCHEMA_REQUESTS.md`.
