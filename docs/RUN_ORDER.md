# Recommended run order (and when to stop)

**Hard deadline:** the ThetaData subscription is to be cancelled by **Nov 10**. Stop downloading when it is
cancelled, then run `python -m playground purge-vendor-data` (dry run) and `--yes` to delete the vendor cache and
write `purge_log.md`. Everything below that needs *new* vendor data must be finished before then; anything that runs
on the cache only (sweeps, stress, management grids) can continue until the purge.

Always `plan` first. `plan` writes nothing and prints, per ticker and per `--entry-cadence`, the requests, rows,
GB and wall-clock at 2 threads (Value tier) and 4 threads (Standard limit), using the latency measured by the last
`probe`. Use `--max-gb` and `--max-hours` (per ticker hard stops) on every `download`.

0. **Setup (day 1).** `probe` (all five tickers) -> paste FAIL lines; `spot` (Alpaca minute bars + coverage);
   `spot --validate massive,greeks,parity` -> read `results/spot_validation.md`; `download --daily-only` (SPY/VIX/VIX3M).
1. **SPY baseline.** `plan --engine lab --tickers SPY --entry-cadences monthly,weekly`, then
   `download --engine lab --tickers SPY --entry-cadence weekly` (vix-distance strikes) and
   `sweep sweeps/spy_core.yaml --limit 1` (the Green baseline row only). Compare with the original engine
   (`backtest`/`matrix`) as a sanity check: same rules, different fill timing (next bar) and fees.
2. **SPY delta sweep (monthly cadence).** `plan --engine lab --strike-mode delta --delta 0.16 --entry-cadences monthly`.
   `download --engine lab --tickers SPY --entry-cadence monthly --strike-mode delta --delta 0.16 --dte-target 45`
   (one bulk snapshot per entry). Repeat per DTE target you want (30/45/60/90: note 60-90 DTE have fewer entry days
   because only monthlies list that far out; skipped days appear as `cadence` in `blocked_entries.csv`).
   Then `sweep sweeps/spy_core.yaml` (cache only). The summary prints the NUMBER of configurations tried and the
   first-half/second-half split for anything that beats the baseline - **read the split, not the ranking**.
3. **SPY management grid on the best few.** Edit the three `config` bundles in `sweeps/spy_management.yaml`
   to the best 2-3 from step 2, download the legs they need (rolls/hedges fetch new legs on demand and cache
   them), then `sweep sweeps/spy_management.yaml`. Use `cf_pnl_to_expiry_per_contract` to see what each rule saved or cost.
4. **Iron butterfly** (and the bull put / broken-wing twins): `sweep sweeps/spy_butterfly.yaml`,
   `sweep sweeps/spy_other_structures.yaml` (monthly cadence; ATM strikes need their own small downloads).
5. **QQQ and IWM**, then **GLD and TLT**: `plan --engine lab --tickers QQQ,IWM` ... `download --engine lab --tickers QQQ,IWM`
   (monthly, delta 0.16 baseline) then `sweep sweeps/other_tickers.yaml`. Check the strike increments `probe` found
   per era; widths are 0.7% of spot rounded to the listed increment. TLT pays monthly dividends: look at `exdiv_flag`.
6. **Stress.** `stress` (or `sweep sweeps/stress.yaml`): forced entries in the approximate windows of
   `data/stress_windows.csv` (edit them; they are month-level, not event dates). Needs the cache to cover the windows.
7. **Export before the purge:** `export-outputs` (derived logs only; never data/, cache/ or parquet), then
   `purge-vendor-data --yes` once the subscription ends.

## Over-fitting guard rails
* A sweep of N configurations will always crown a winner. Prefer few, pre-registered configurations; the summary
  prints N and flags anything better than baseline together with its first/second-half return on risk.
* Prefer return on risk and the half-vs-half columns to total P&L; zero-trade configurations are excluded from rankings.
* Real-quote evidence is ~10 years at most (quotes verified back to 2016-10-14); treat results as a tier-1 study.
