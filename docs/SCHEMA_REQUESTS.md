# Schema requests from the reporting layer (TASK B) to the engine (TASK A)

The reporting layer works with the trade-log schema exactly as given. These are optional additions; none is required.

1. **`stop_level_per_contract`** (trades.csv, USD, positive number = the loss level that triggers the stop).
   Slippage-through-stop in the stress lab needs the trigger level. Until the column exists the report derives it from
   `config.json`: `stop_dollar` (USD per contract) or `stop_multiple` (x credit) plus an optional `stop_floor_width_frac`
   (floor = frac x width x 100). With neither, the slippage column prints "n/a (no stop level)".
2. **`config.json` key `synthetic`** (bool). The generator writes `true`; real runs should omit it or write `false`.
   Any run with `synthetic: true` makes the whole report carry the SYNTHETIC banner.
3. **Stress runs**: name the profile `stress_<window>` (e.g. `stress_covid2020`); experiments `experiment_<name>`
   (`experiment_no_stop`, `experiment_all_off`). The "ways to lose" matchers live in `config/ways_to_lose.yaml` (regexes) so
   they can follow whatever naming the engine actually uses.
4. **`cf_pnl_to_expiry_per_contract`** should be net of the same fees as `pnl_per_contract` (the scorecard compares them directly).
5. **Same entries across configs**: management / delta comparisons pair on (ticker, strategy, entry date). Keep entry
   timestamps identical across management configs run on the same entry set.
