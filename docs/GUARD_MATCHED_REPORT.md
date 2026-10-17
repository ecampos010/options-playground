# Matched guarded vs plain comparison: method, audit and VPS run

**Status: code and tests done. No real-data results yet.** The comparison is built to run on the VPS against the
existing ThetaData cache. Nothing was downloaded here. The only run in this session was a synthetic plumbing check
(2016-01 to 2016-06, synthetic prices and calendar), and its numbers mean nothing.

Question it answers: how much of a Green-rule result comes from the guards? Same entry days, same exits, guards on
vs off, on real SPY quotes, 2016-10-01 to 2026-09-30.

## 1. Baseline profile audit: `profiles/experiment_all_off.yaml`

The profile sets six keys (`gate`, `upside_guard`, `event_buffer`, `cb_days`, `earnings_filter`, `term_filter`).
Every entry guard Green uses in `playground/rules.py` is off in it.

| Guard (Green code) | Green default | `experiment_all_off` |
|---|---|---|
| VIX gate: prior-day VIX close >= 25 | on (`gate_vix` 25) | off |
| Day move: SPY day change <= -1.5% (inside the gate) | on (`gate_day`) | off (same switch as the gate) |
| 5-day move: SPY 5-day change <= -4% (inside the gate) | on (`gate_5d`) | off (same switch as the gate) |
| Up-move guard: SPY 5-day change >= +4% | on (`upside_5d` 0.04) | off |
| Day-up move >= +1.5% | **not in the engine** | n/a |
| Event buffer: FOMC, CPI, PPI, NFP, ELECTION, entry 0-2 trading days before (day-of included) | on | off |
| Circuit breaker: 21 trading days after a losing month | on (`cb_days` 21) | off (0) |
| Earnings filter (SPY has no EARNINGS rows, so inert) | on | off |
| Term filter (VIX/VIX3M > 1) | off in Green | off |
| Post-spike wait, min IV rank, delta cap, max open lots, Red's gate-trip requirement, force-entry | off in Green | off |

**Answer: `experiment_all_off` turns off every entry guard that exists in Green.** No entry filter remains active.
Two differences from Green remain:

1. **Budget.** The profile's `tier` is `experiment`, which maps to a 5% budget (`TIER_BUDGET` in `config.py`). Green's
   tier is 15%. So `all_off` is not a clean guard-off baseline: it also trades a third of the size. `ladder
   --equal-budget` fixes this for the ladder, but the existing all-off runs did not use it.
2. **Sizing caps stay on** (per-expiry cap at 60% of budget). These limit size; they are not entry guards.

Where the profile is used:

- `reporting/ways.py:107`: the "All guards off" row of the ways-to-lose table (matched by `config/ways_to_lose.yaml:6`).
- `reporting/synth.py:364`: a synthetic reporting variant.
- `reporting/tests/test_playbooks.py:44,55`.
- `tests/test_lab.py:281,290`: asserts `gate` is False and `cb_days` is 0.
- `sweeps/stress.yaml:8`: stress sweep. Stress forces entries, so the filters are moot there.
- `playground/ladder.py:17`: the ladder's experiment list.
- `docs/SCHEMA_REQUESTS.md:12`: documentation.

**Not found:** `claude/alpaca-state.md` is not in the repo. The only file under `claude/` is
`claude/guard-calibration-study.py.txt`. It defines a candidate "MOVE GUARD" (day >= +1.5% or |5-day| >= 4%) for a
study, not for the engine. Green's rules are defined in `playground/config.py` (defaults), `profiles/green.yaml`
and the README.

## 2. The two arms (`playground/guard_matched.py`)

Both arms: lab engine, SPY, iron condor, delta 0.16 on both short strikes, DTE target 45 (the setting the existing
lab runs use, `docs/RUN_ORDER.md` step 2), 5-minute resolution, natural bid/ask fills, 0.05 USD per contract per
leg, Green's 15% budget, equity-pct sizing, same entry-window and seed. Cadence: monthly and weekly, run separately.
Window: 2016-10-01 to 2026-09-30.

| | PLAIN | GUARDED |
|---|---|---|
| Entry guards | none | gate, upside, event buffer, circuit breaker, earnings (inert for SPY) |
| Exits | stop 2x credit on natural buyback; take-profit 50% of credit; breach; 21-DTE time exit | same |
| Entry days | every scheduled day | only the days PLAIN entered |

**Green rules applied in the guarded arm (exactly):**

1. Gate: prior-day VIX close >= 25, or SPY day change (entry-bar spot vs prior close) <= -1.5%, or 5-day change
   (entry-bar spot vs close 5 trading days back) <= -4%.
2. Up-move guard: 5-day change >= +4%.
3. Event buffer: no entry on an FOMC, CPI, PPI, NFP or ELECTION day, or on the 1 or 2 trading days before it.
   The release-time exception is off in Green.
4. Circuit breaker: no entries for 21 trading days after a calendar month that ends below its start equity.
5. Earnings filter: on, but SPY has no EARNINGS rows, so it never fires.

**Not applied:** the day-up half of the move guard (>= +1.5%), because the engine does not implement it.

## 3. How the matching works

- PLAIN runs first. Its entry dates become a whitelist for GUARDED (`run_lab(..., entry_days=...)`, new in
  `playground/lab.py`). GUARDED can therefore enter only days PLAIN entered. The subset is enforced by construction
  and checked by `tests/test_guard_matched.py::test_guarded_entries_are_a_strict_subset_of_plain_entries`.
- Strike, expiry, and entry bar are functions of the day and the market, not the book, so they are identical in both
  arms. Exits depend only on each position's own quotes, so they are identical too.
- Each arm keeps its own cash, sizing and equity path. **The return on risk per trade is sizing-free and is the
  primary metric.** Dollar figures and drawdown include the sizing-path effect and are labelled that way.
- Each skipped trade is attributed to the guard that blocked its entry day in the guarded arm.
- Split sample: first half vs second half at the calendar midpoint of the window (2021-09-30). The request was cut
  off at "first half vs. second half of", so the split is my choice.

## 4. Decisions and deviations to review

1. **Take-profit is not at mid.** You specified "take-profit 50% of credit (mid)". The engine triggers take-profit
   and realizes it at `cfg.fill`, which is natural in Green, so both arms use natural. Stops are natural, as
   specified. Mid take-profit needs a separate take-profit fill in the shared engine. I did not make that change
   because it touches every run ID's inputs. It is a small follow-up if you want it.
2. **Day-up move half of the move guard: not applied** (see section 2).
3. **Circuit breaker is in the guarded arm** (it is a Green default, though not in your list). Its blocks are
   labelled `cb` in the reason tables, so you can read the result with or without it.
4. **Budget is 15% in both arms**, not the `experiment` 5%.
5. **Matched, not native.** A native guarded run would also free capacity that PLAIN never used, and could enter
   extra days. That effect is not in this comparison. If you want it, it is a third arm and can be added later.
6. **Cache gaps are now counted.** Before this change, a leg that was missing from the cache was silently treated as
   "not monitored" for that day (no stop, no exit), with no count. `_Lab._load` now counts those position-days in
   `meta.json` (`leg_load_gaps`), and the report shows them per arm. A non-zero count on PLAIN means a result cannot
   be trusted yet.

## 5. VPS run

**Pre-requisites.** `data/` must already contain SPY daily and VIX daily, 5-minute SPY spot (`spot`), real
`data/events.csv` (`scripts/fetch_context_data.py`), and option quotes for the PLAIN arm's entries.

**The option-quote download must run with the guards off.** Stage A of the downloader pre-filters entry days with the
guard rules (`playground/downloader.py` `candidate_days`), and Stage B runs the engine with the same flags. Run with
the guards on, it would fetch quotes only for days GUARDED would enter. PLAIN would then have no data on its
guard-blocked days, and the comparison would be silently wrong. Use these flags for both cadences:

```bash
python -m playground plan     --engine lab --tickers SPY --strike-mode delta --delta 0.16 --dte-target 45 \
       --entry-cadences monthly,weekly --gate off --upside-guard off --event-buffer off --cb 0 --resolution 5m
python -m playground download --engine lab --tickers SPY --strike-mode delta --delta 0.16 --dte-target 45 \
       --entry-cadence monthly --gate off --upside-guard off --event-buffer off --cb 0 --resolution 5m \
       --max-gb 20 --max-hours 12          # then the same with --entry-cadence weekly
```

Check the `plan` output and the download manifest. The subscription ends **Nov 10** (`docs/RUN_ORDER.md`), so the
download must finish before then. The comparison itself reads only the cache and never calls ThetaData:

```bash
python -m playground.guard_matched --provider theta --data-dir data --events data/events.csv \
       --start 2016-10-01 --end 2026-09-30 --cadence monthly weekly --out results/guard_matched
```

Output per cadence in `results/guard_matched/<cadence>/`: `summary.md` (all tables), `summary.json`,
`trades_plain.csv`, `trades_guarded.csv`, `skipped_trades.csv`, `blocked_plain.csv`, `blocked_guarded.csv`.
`results/` is git-ignored.

**Before trusting any number,** check that PLAIN shows zero gap days and zero `position-days not monitored`. The
guarded arm can only have a subset of these, so checking PLAIN is enough. A non-zero count means the download was
incomplete. Re-run the download, do not interpret the table.

## 6. What was verified here

- `pytest`: 308 passed, 2 skipped on the current `main` (after PR #9; before it, 279 passed, 2 skipped, against a 269 baseline). The reporting suite also passes (83 passed).
  The 10 new tests are in `tests/test_guard_matched.py`. They cover: the arms differ only in the guard switches; the guarded arm carries
  Green's guard values; both arms share the budget, strikes and exits; the strict subset holds on a market where
  guards trip and an FOMC date is present (monthly and weekly); the whitelist leaves an arm without guards unchanged;
  run IDs of existing runs are unchanged; a cache gap is counted; and the metrics on a hand-computed path (return on
  risk, worst 5%, drawdown, worst cycle, dollars).
- Synthetic CLI run (plumbing only): produces all outputs, and the per-reason and split-sample tables reconcile with
  the totals. Its numbers are not results.
- Not verified: anything on real data. Also unverified per the README: the Alpaca 5-minute spot path, so check the
  `spot` validation output before the run.

## 7. Results

Pending the VPS run. This report contains no results, and no synthetic numbers should be quoted as results.

## 8. Changed files (uncommitted, in the working tree)

- `playground/lab.py`: optional `entry_days` argument on `run_lab` (default `None`, existing behavior unchanged);
  `leg_load_gaps` counter in `meta.json`. The run-ID hash keeps its old inputs unless `entry_days` is used.
- `playground/guard_matched.py`: new. The two arms, the matched run, metrics, split sample, report and CLI.
- `tests/test_guard_matched.py`: new, 10 tests.
- `docs/GUARD_MATCHED_REPORT.md`: this report.

Nothing is committed or pushed. The repo is attached read-only, so the VPS needs these files delivered by a push
(which needs push access) or by copying the working tree.
