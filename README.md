# options-playground

ThetaData-backed backtester for the **Green** SPY iron-condor rules, with a probe, a tight two-stage
downloader, a plan (dry-run) command, a rule matrix and the traffic-light **ladder**
(Green / Yellow E1-E7 / Red). Everything runs offline against a built-in **synthetic** provider.

> **Status: the ThetaData adapter is partly verified** (see "Verified vs unverified" at the bottom: lists and
> single-contract 1m quotes work on a Standard account; bulk and intraday-underlying are open). Re-run `probe` on
> the VPS after pulling and paste back any FAIL lines. Synthetic results say nothing about
> real markets (the synthetic world is only there to exercise the code).

License: placeholder - the owner will choose a license later. No third-party code is copied in.

## VPS quickstart (Linux, Python 3.10+)

```bash
git clone <repo> && cd options-playground
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # optional: pip install yfinance (SPY daily, tried first)
cp .env.example .env                       # placeholders only; .env is git-ignored. Never commit credentials.
set -a; source .env; set +a                # THETA_USER / THETA_PASS (+ THETA_BASE, default http://127.0.0.1:25503)
# start Theta Terminal (v3) with those credentials, see ThetaData docs, e.g. java -jar ThetaTerminal.jar
python -m pytest -q                        # sanity (offline)

python -m playground probe                 # -> results/probe_report.md (+ .json). Paste FAIL lines back.
python -m playground download --daily-only # SPY daily (yfinance, else Theta) + VIX daily (Cboe CSV) -> data/
python scripts/fetch_context_data.py       # FOMC/CPI/PPI/NFP/ELECTION -> data/events.csv, fed funds -> data/rates.csv (needs internet)
python -m playground plan --start-date 2016-10-01 --end-date 2026-09-30 --resolution 5m
python -m playground download --start-date 2016-10-01 --end-date 2026-09-30 --resolution 5m \
        --threads 2 --max-rps 5 --cover-matrix       # resumable: stop (Ctrl-C) and re-run any time
python -m playground backtest --profile green
python -m playground matrix                # 4 guard sets x 2 exits (8 runs), compact table
python -m playground matrix --fill-compare # natural vs mid side by side
python -m playground ladder --split-sample
```

Offline demo (no ThetaData needed): add `--provider synthetic` to `backtest`, `matrix`, `ladder`, `plan`
(`probe --fake` / `download --fake` use an offline *simulator of the assumed API shapes*, plumbing test only).

## Commands

| command | purpose |
|---|---|
| `probe` | (a) expired SPY contracts listable ~1/3/5/8/10y ago, (b) bid/ask+sizes at 1m and finer, (c) size/latency of a ~40-strike day at 1m (+ bulk `strike=*` test), (d) request-rate burst / 429 behaviour, (e) SPX/VIX index history and VIX options. Fails soft: prints endpoint, params, status, raw error. |
| `plan` | candidate days, expected trades, requests/rows/GB at 1m/5m/15m/eod, wall-clock at 2 and 4 threads (latency from the last probe), plus the naive full-chain cost. Writes nothing to `data/`. |
| `download` | daily data, then Stage A (narrow entry snapshot: up to 3 trial-confirmed expiries nearest the target DTE, Fridays first, +/-10 listed strikes around the rule strikes, at the entry bar) for every candidate day, then Stage B (the engine runs in fetch mode and pulls full-life quotes for only the 4 chosen legs, in `--chunk-days` pieces). Parquet in `data/options/res=<r>/date=<d>/expiry=<e>/right=<P|C>/`; coverage is recorded so re-runs fetch nothing. `--max-rps`, exponential backoff on 429/5xx, `--entry-fetch bulk` (one `strike=*` request per expiry; only if probe (c) shows it works), prints a manifest. `--cover-matrix` makes the cache cover all 8 matrix variants. |
| `backtest` | one run; prints <=40 lines; writes `results/<run_id>/{trades.csv,equity.csv,config.json,summary.md}` |
| `matrix` | standard 8-run grid (+ `--fill-compare`) |
| `ladder` | green + yellow_E1..E7 + red on the same data/seed/fill; `--split-sample`, `--equal-budget` (give everyone Green's budget to isolate the guardrail effect) |
| `python -m reporting report` | portfolio report; now also cash interest from `data/rates.csv` (with / without interest and vs cash) and the month / drawdown "why" cards with a Red months table: see `reporting/README.md` |
| `overlay`, `spike-study` | **reserved stubs** (see `playground/reserve.py`; the engine's Position = signed legs + `kind`, so a hedge leg can be added without a rewrite) |

Every rule is a config default overridable by CLI flag: `--gate on|off --upside-guard on|off --event-buffer on|off
--cb 0|21 --tp --stop-mult --time-exit --fill natural|mid|haircut --haircut --entry-window 09:45-15:30
--entry-sample first|random|every-30m --seed --start 100000 --apy --budget-pct --full-use --strike-mode vix|delta
--short-delta --dte 30,35,45 --min-credit --expiry-cap-frac --resolution --profile ...` (`backtest --help`).

## events.csv and rates.csv (context data)

```bash
python scripts/fetch_context_data.py                    # 2016-2026 -> data/events.csv + data/rates.csv; run on the VPS (internet)
python scripts/fetch_context_data.py --check            # re-validate the files already in data/ (no network)
python scripts/fetch_context_data.py --html-dir saved/  # parse pages you saved in a browser if a site blocks scripts
python scripts/fetch_context_data.py --allow 2025:CPI=11   # accept a genuine disruption, printed and never silent
```
Sources, in the order tried: **FOMC** federalreserve.gov `fomccalendars.htm` plus `fomchistorical<year>.htm` for 2016-2020 (unscheduled / emergency meetings such as March 2020 are included as FOMC rows; the ~8 check counts scheduled ones; notation votes are skipped); **CPI / PPI / NFP**
(Employment Situation) bls.gov archived releases `bls/news-release/<cpi|ppi|empsit>.htm` (link names embed the date), then the schedule page
`schedule/news_release/<slug>.htm`, then the monthly lists `schedule/<year>/<MM>_sched_list.htm` (recent years only); **fallback** FRED `fred/release/dates`
(releases 10 CPI, 46 PPI, 50 Employment Situation) when `FRED_API_KEY` is in your environment (`set -a; source .env; set +a`; never printed or committed);
**ELECTION** from the statutory rule; **rates** the FRED public CSV (`--rate-series DFF|EFFR`, no key).
It never aborts on one bad source: it writes what it found (a previous good file fills holes), prints a coverage table (event x year: count or MISSING), which
source each event/years used and the non-fatal fetch failures. Exit code 0 only if FOMC, CPI and NFP exist for every year, else 4 (partial files still written);
count anomalies (not ~8 FOMC / 12 CPI/PPI/NFP) print as WARNING, and `--strict` turns them into exit 2. Weekend dates are dropped as parse errors.
The URL patterns were taken from web search results because the cloud sandbox cannot reach the sites; the parsers are tested on saved fixtures only.

`data/events.csv`, columns `date,event[,symbol]`; event in `FOMC, CPI, PPI, NFP, ELECTION, EARNINGS`
(`symbol` only matters for EARNINGS). Lines starting with `#` are ignored. See `templates/events.csv.template`.
`data/rates.csv` is `date,rate_pct` (percent, calendar-daily). **This repo never invents dates.** A missing file is a clear error (use
`--event-buffer off` to run without it). With `--provider synthetic` (or `--fake`) a clearly-labelled SYNTHETIC
event calendar is ALWAYS used, even if `data/events.csv` exists: real dates against invented prices say nothing (and, with the real calendar, the
synthetic GLD sweep loses its only two entries to CPI/PPI windows).

## Rule interpretations I had to choose (check these)

* Entry cadence is not in the brief: the engine tries one entry per trading day (`--entry-every N`) and fills
  whatever budget headroom exists; per-expiry cap and budget limit it. Equity = cash + mid-marked open positions.
* Gate/guards use **prior-day VIX close**, SPY spot at the entry bar vs prior close (day change) and vs close 5
  trading days earlier (5-day change). No look-ahead.
* "Within 2 trading days before" an event blocks offsets 0,1,2 (event day included). E1 allows day-of entries after
  the release time; E2 blocks FOMC day-of only.
* Stop uses the NATURAL buyback; take-profit and realised exit price use the selected fill model. Exit order per bar:
  stop > breach > take-profit > time. Time exit only applies if entry DTE > `--time-exit` (so E3 has none).
* Circuit breaker: if a calendar month ends below its start equity, no entries for 21 trading days.
* Interest: simple daily accrual on the whole cash balance (`apy/365` per calendar day). Commission default 0.
* Red = gate switched off + `entry_requires_gate_trip` + one open lot + 1% budget. Optional per-flag ignores:
  `--red-ignore event,expiry-cap,time-exit,stop`.
* Yellow profiles use the Yellow budget (5%); Green 15%; Red 1% (profile `tier`). Compare guardrail effects
  cleanly with `ladder --equal-budget`. E5 (earnings) and E6 (concentration) are implemented but N/A for SPY-only.
* E3 (7-14 DTE) with the unchanged VIX-distance strike rule and 0.50 minimum credit may produce almost no trades.
* Stage A is intentionally per-strike by default (slow but least assumption-heavy); bulk is opt-in.

## Lab engine: multi-ticker, multi-strategy, management rules (TASK A)

The original Green engine (`backtest`, `matrix`, `ladder`) is unchanged. The **lab engine** adds five underlyings,
delta strikes, six defined-risk strategies, a management-rule library and standardized trade logs that the
reporting/portfolio layer consumes (`docs/TRADE_LOG_SCHEMA.md` is the contract; sample in `tests/fixtures/sample_run/`).

```bash
python -m playground spot --tickers SPY,QQQ,IWM,GLD,TLT --validate massive,greeks,parity   # Alpaca minute bars (UNVERIFIED)
python -m playground plan --engine lab --tickers SPY,QQQ --strike-mode delta --delta 0.16 --entry-cadences monthly,weekly
python -m playground download --engine lab --tickers SPY --entry-cadence monthly --delta 0.16 --max-gb 20 --max-hours 12
python -m playground run --ticker SPY --strategy iron_condor --delta 0.16 --dte-target 45 --entry-cadence monthly \
       --trigger distance --trigger-value 0.02 --action roll_tested --max-rolls 2        # unit mode by default
python -m playground sweep sweeps/spy_core.yaml [--limit N]     # resumable; prints configs tried + half-vs-half
python -m playground stress --only covid_crash_2020             # forced entries, approximate month-level windows
python -m playground ladder                                      # now includes experiment_* profiles + Level column
python -m playground export-outputs                              # derived logs only (never data/, cache/, parquet)
python -m playground purge-vendor-data [--yes] [--include-spot]  # dry run unless --yes; writes purge_log.md
```
Offline demo: add `--provider synthetic` to `run`, `sweep`, `stress`, `plan`. Run order and the Nov 10 stop:
`docs/RUN_ORDER.md`.

| area | flags |
|---|---|
| universe | `--ticker SPY/QQQ/IWM/GLD/TLT` (`profiles/tickers.yaml`), `--width-mode auto/pct/fixed`, `--width-pct 0.7%`, `--min-credit-pct 10%` |
| strikes | `--strike-mode vix-distance/delta`, `--delta`, `--put-delta`, `--call-delta`, `--dte-target 30/35/45/60/90`, `--entry-cadence monthly/weekly/twice-weekly/daily`, `--scale-in 1/4` |
| strategies | `--strategy iron_condor/iron_butterfly/bull_put_spread/put_bwb/cash_secured_put/covered_call` (undefined-risk structures are not implemented) |
| exits | `--tp`, `--ifly-tp 25/50`, `--stop-mult 1.5/2/3/off`, `--stop-dollar`, `--time-exit 0/21/7`, `--breach on/off` |
| triggers | `--trigger distance/delta --trigger-value ...  --action none/exit/roll_tested/roll_condor/roll_untested/close_tested_spread/hedge_put_call/hedge_shares`, `--max-rolls`, `--roll-credit-only`, `--roll-to-pct`, `--hedge-delta`, `--hedge-expiry`, `--stock-slip-bps` |
| entry filters | `--term-filter on` (VIX/VIX3M > 1), `--post-spike-wait N`, `--min-ivr 0/30/50`, `--delta-cap 150/200`, existing gate/upside/event/cb |
| fills / fees | next-bar natural fills for every trigger/stop/exit; `--fee-per-contract 0.05` per contract per leg |
| spot | `--spot-source alpaca/parity/daily`, `--spot-fallback parity/none` (missing days are never interpolated) |
| sizing | `--sizing equity-pct` (budget % of current equity) or `fixed1` (unit mode: 1 contract, no budget, circuit breaker off) |

`--spot-source daily` enters at the 16:00 close and checks breaches/triggers on daily bars only (a result always exists, but it is a coarser study).

Interpretation notes (check these): a roll targets a short strike at `spot*(1-+roll_to_pct)` (default 1%), required to be
closer to spot than the old short; `roll_condor` returns both shorts to their ENTRY distances from the current spot (both
follow the move); stop = loss >= (N-1) x credit on the NATURAL close cost; counterfactual = original structure's expiry
payoff from the settlement-day close (identical to quotes at expiry, needs no post-exit data); `hedge_cost_per_contract`
= option premium or share slippage paid, the hedge P&L itself is inside `pnl_per_contract`; gate day/5-day moves use the
traded underlying's own spot (VIX/VIX3M are global); a trade may lose slightly more than `collateral_per_contract` through
natural exit spreads (the validator allows max($25, 5%)).

## Verified vs unverified (ThetaData adapter, original engine)

**Verified on a real ThetaData STANDARD options account** (Terminal v3 at 127.0.0.1:25503, run on a VPS):
* `option/list/expirations` (SPY: 2,148 expirations 2012-06-01..2029-01-19) and `option/list/strikes`: HTTP 200.
* `option/history/quote` (single contract, 1m): columns `bid_size, bid_exchange, bid, bid_condition, ask_size,
  ask_exchange, ask, ask_condition` + a timestamp column (parser accepts several names). Real bid/ask/sizes seen for
  SPY on 2023-10-13 (exp 20231117, K414), 2021-10-15 (exp 20211119, K428) and 2016-10-14 (exp 20161118, K205).
* **HTTP 472 "No data found" (empty body) = contract not listed/tradable that date -> NO_DATA**, not an error: never
  retried, cached as a negative, counted separately (`no_data_472` in the manifest).
  Seen: on 2025-10-15 only Friday expirations answered 5 weeks out (Mon-Thu were in the list but gave 472); on
  2018-10-15 half-dollar strikes (e.g. 263.5) gave 472 while integer strikes returned rows.
* Stock and index history are HTTP 403 (Stock/Index = FREE tier). Reported by `probe` as INFO, never a failure:
  VIX comes from the Cboe CSV and SPY daily from yfinance.

**Contract selection (implemented from those facts, tested only against an offline simulator of them):**
expiries in the DTE window are ranked Friday/monthly-first and each is *confirmed with a trial 1-minute quote* at the
entry minute (keep only those returning rows; result cached per date/expiry, so re-runs repeat nothing). Strikes come
from the strikes list; one probe per date decides whether half-dollar strikes traded (`data/listing/trial/`), and the
engine only ever picks among strikes that actually returned quotes, nearest to the rule target.

**Still unverified:** the bulk entry form (`strike=*`, `right=both`, `start_time/end_time`; `probe` section c2 reports
rows/bytes/latency/columns or the exact error), option EOD (`--resolution eod` quotes), finer-than-1m intervals,
real request-rate limits (Standard: 4 concurrent; Value: 2 - `plan` shows both), Cboe CSV URL, and any real-market
strategy result.

**Known gap (decision needed):** intraday SPY bars cannot come from ThetaData on this plan (stock history 403). The
engine needs a spot at each bar for entry strikes, the gate's day/5-day moves and breach exits, so 1m/5m/15m runs on
real data need another spot source. `--resolution eod` is self-contained (spot = daily close). Stage A tolerates the
missing bars (entry minutes come from the grid) but backtests will report data gaps until a source exists.

Verified offline (`pytest`, 59 tests): sizing, gate/guards/event/circuit breaker, fills, exits, interest, caps,
determinism, ladder single-switch profiles, cache coverage/idempotent re-runs, 472/403 handling, Friday-first
fallback and half-dollar handling, probe/plan plumbing.


### Verified vs unverified: lab engine additions

| piece | status |
|---|---|
| strategies, payoff/max-loss math, management rules, counterfactuals, trade-log schema, sweep/stress/export/purge | tested offline (synthetic provider + hand-built markets) |
| Alpaca spot path (`spot`: sip 1Min bars, adjustment raw/all, pagination, corporate actions) | **UNVERIFIED** - endpoint/params/response shape from memory; run `spot` on the VPS |
| spot cross-checks (Massive minute bars, Theta Greeks `underlying_price`, put-call parity) | **UNVERIFIED** - Massive base URL/fields, Greeks endpoint/columns; stats land in `results/spot_validation.md` |
| Greeks endpoint (`delta`, `implied_vol`, `underlying_price`) | **UNVERIFIED** - `probe` section f reports the fields; absent -> Black-Scholes delta from mid IV (logged in `meta.delta_source`) |
| bulk chain snapshot (`strike=*`, `right=both`, one minute) | **UNVERIFIED** - `probe` sections c2/f report rows/latency or the exact error |
| VIX3M / VXN / RVX / GVZ Cboe CSVs | **UNVERIFIED** URLs (VIX3M needed only for `--term-filter`; others skipped silently) |
| ex-dividend dates (Alpaca corporate actions -> yfinance -> none) | **UNVERIFIED**; missing data prints a warning and flags stay 0 |
| per-ticker option listings, strike increments by era, VIX options | **UNVERIFIED** until `probe` runs on the VPS for all five tickers |
| delta-mode download volume/time | estimates only (`plan`); nothing measured on real data |
