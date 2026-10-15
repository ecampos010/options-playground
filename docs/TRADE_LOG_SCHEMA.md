# Trade-log schema (contract between the backtest layer and the reporting/portfolio layer)

Produced by `python -m playground sweep|run|stress`. Validated by `playground/schema.py`
(`validate_run_dir`). **Do not rename columns without updating this document and `schema.py`.**
A tiny conforming sample lives in `tests/fixtures/sample_run/`.

Conventions: all money is **USD per ONE contract** (the 100 multiplier is applied). Timestamps are
US/Eastern, ISO 8601 (`2024-03-05T10:15:00-05:00` or naive local ET — consumers must treat both as ET).
Dates are ISO 8601 (`2024-03-05`). Blank = empty CSV cell. CSV files have a header row and use `,`.

Licensing: these files contain **only derived per-trade numbers** — never raw bid/ask series.

## runs/<run_id>/trades.csv  (one row per closed position; unit mode = 1 contract)

| column | type | meaning |
|---|---|---|
| run_id | str | directory name; unique per configuration (config hash) |
| trade_id | str | `<run_id>-<n>` unique within a run |
| ticker | str | SPY, QQQ, IWM, GLD, TLT |
| strategy | str | iron_condor, iron_butterfly, bull_put_spread, put_bwb, cash_secured_put, covered_call |
| level | int | 1 Green, 2 Yellow, 3 Red, 4 Experiment |
| profile | str | profile name (green, yellow_E4, red, experiment_*, stress_<window>) |
| config_tag | str | short human-readable parameter summary (e.g. `d16_dte35_tp50_sl2_m`) |
| entry_ts, exit_ts | ISO ET | fill times (next-bar fills for any trigger/stop/exit) |
| expiry | date | expiry of the original structure |
| dte_entry | int | calendar days to expiry at entry |
| exit_reason | enum | `tp, stop, stop_dollar, breach, trigger_exit, time, expiry, other` |
| width | USD/share | spread width (strike distance; BWB: the near-wing width) |
| collateral_per_contract | USD | exact max loss of one contract at entry (CSP/CC: strike×100 / spot×100) |
| credit_per_contract | USD | net credit received at entry, natural fill (negative = debit) |
| pnl_per_contract | USD | realised P&L net of fees and all adjustment/hedge costs |
| fees_per_contract | USD | all fees paid over the life (`--fee-per-contract` × leg-contracts × transactions) |
| adj_count | int | number of adjustments (rolls, spread closes) |
| adj_cost_per_contract | USD | net cost of adjustments (positive = paid; excludes fees) |
| hedge_cost_per_contract | USD | hedge premium paid (options) or slippage paid (shares); the hedge P&L itself is inside `pnl_per_contract` |
| cf_pnl_to_expiry_per_contract | USD | P&L the ORIGINAL structure would have had if held to expiry (net of its entry fees); blank if the trade ran to expiry unadjusted or data are missing |
| entry_spot, exit_spot | USD | underlying price at fill (exit_spot = settlement close at expiry) |
| entry_vix | float | VIX close of the previous trading day (no look-ahead) |
| vix_bucket | enum | `lt15, 15to20, 20to30, gt30` |
| vix_term_ratio | float | VIX / VIX3M (prior close); blank if VIX3M unavailable |
| short_put_delta, short_call_delta | float | delta of the short legs at entry (blank if absent) |
| exdiv_flag | 0/1 | 1 if a short call was ITM within 2 trading days of an ex-dividend date at any time |
| guards_on | str | `;`-separated enabled guards, e.g. `gate;upside;event;cb` (unit mode: cb is off) |
| fill_model | str | natural, mid, haircut |
| spot_source | str | alpaca, parity, daily, synthetic — the source actually used at entry |

## runs/<run_id>/marks.csv

`trade_id, date, mark_pnl_per_contract` — one row per trading day each position was open (entry day through
exit day). Value = end-of-day liquidation value (natural basis: shorts bought at ask, longs sold at bid)
plus cash already collected, minus fees so far, in USD per contract. The exit-day mark equals the realised P&L.
Days without a valid quote carry the previous mark.

## runs/<run_id>/blocked_entries.csv

`date, ticker, reason` with reason in `gate, upside, event, cb, term, spike, ivr, nodata, noquote, mincredit, cadence`.
Extension values (treat unknown values as `other`): `deltacap` (net-delta cap), `size` (no budget headroom,
equity-pct mode only). `cadence` is logged only for days where the cadence expected an entry but no expiry
near the DTE target was listed.

## runs/<run_id>/config.json, meta.json

* `config.json`: `flags` (every setting), `git_commit`, `data_range` (first/last trading day backtested),
  `window` (stress window dates or null), `command`.
* `meta.json`: `counts` (trades, exit reasons, blocked by reason), `versions` (python, pandas, numpy, playground),
  `spot_coverage` (days with full coverage / flagged / fell back), `delta_source_tag` (`theta_greeks` | `bs_fallback` | `mixed`), `delta_source` (provider vs BS-from-mid
  counts), `unit_mode` (bool), `notes`.

## Unit mode vs equity mode

`--sizing fixed1` (unit mode) is what B consumes: exactly 1 contract per entry, no compounding, no budget cap,
circuit breaker off. `--scale-in 4` in unit mode emits four independent 1-contract rows per cycle whose
`config_tag` contains `scalein4`; a consumer should weight each at 25% when sizing.
Equity mode additionally writes `equity.csv` and `sizing.csv` (`trade_id, contracts, equity_at_entry`).
