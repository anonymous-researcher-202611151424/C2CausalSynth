# Reviewer guide

Start with `README.md` for installation. The following checks have different
purposes and do not require access to operational records.

1. **Inspect the simulator.** Run `examples/quickstart.py`, then inspect
   `src/c2csynth/item_market.py`. Parameters, coupon validity, and initialization
   are specified in the single reference-world directory.
2. **Inspect the reported numbers.** Run `reproduce/verify_results.py` and
   `reproduce/figures.py`. These use the frozen log-level/population-level
   aggregates and regenerate the manuscript's tables and figures.
3. **Exercise the complete learning/evaluation path.** Run the `smoke`
   benchmark. For all three assignment regimes and shared-oracle reuse, run
   `python -m c2csynth.suite --preset smoke --output outputs/smoke-suite`.
4. **Rerun the reported synthetic experiment.** Use the `paper` suite. It
   generates 42-day factual logs, trains the reported estimators, and scores
   cumulative effects at days 7, 14, and 21. Full settings are in
   `docs/benchmark.md`.

`docs/data.md` specifies the 11 released initial-state columns and fixed
splits. `docs/fidelity.md` specifies the empirical fitting/evaluation method
and the scope of public reproduction. `results/release_validation.json`
records the export checks, including comparisons against the retained
experiment outputs. These checks establish continuity of the released
implementation; they are not a new empirical validation study.

For custom experiments, supply a new initial table and explicit parameters
to `DailyItemMarket`. Preserve the frozen reference resources when reproducing
the paper; their loader rejects changes against the recorded hashes.
