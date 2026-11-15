#!/usr/bin/env python3
"""Collect the item-market benchmark over logging regimes and log replicates.

Reads ``<root>/<regime>/log*/reports/horizon_comparison.csv``, one directory per
factual log, and writes to ``<root>/summary``:

- ``per_log.csv``: one estimate per regime, log, estimator, policy, horizon and
  outcome (several learning seeds on one log are averaged first);
- ``summary.csv``: over logs, the mean estimate, its bias against the oracle,
  the spread between logs, the errors, and the number of logs on which the
  estimate is closer to the oracle than a prediction of no effect.

All units are cumulative event-days per 100 initial users. The printed table
shows the longest horizon.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["policy", "reference_policy", "horizon", "outcome"]


def collect(root):
    frames = []
    for path in sorted(root.glob("*/log*/reports/horizon_comparison.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "regime", path.parents[2].name)
        frame.insert(1, "log", path.parents[1].name)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No completed reports under {root}")
    runs = pd.concat(frames, ignore_index=True)
    # Every log is scored against the same oracle draws of the same evaluation users.
    spread = runs.groupby(KEYS).oracle_effect_per100.agg(lambda values: values.max() - values.min())
    if spread.max() > 1e-9:
        raise SystemExit("Reports were scored against different oracles")
    return runs


def summarize(runs):
    per_log = runs.groupby(["regime", "log", "model", *KEYS], as_index=False).agg(
        runs=("run", "size"), oracle_effect_per100=("oracle_effect_per100", "first"),
        oracle_mc_se_per100=("oracle_mc_se_per100", "first"),
        predicted_effect_per100=("predicted_effect_per100", "mean"))
    per_log["error_per100"] = per_log.predicted_effect_per100 - per_log.oracle_effect_per100
    per_log["beats_zero"] = per_log.error_per100.abs() < per_log.oracle_effect_per100.abs()
    rows = []
    for key, group in per_log.groupby(["regime", "model", *KEYS], sort=False):
        error = group.error_per100.to_numpy()
        oracle = group.oracle_effect_per100.iloc[0]
        spread = error.std(ddof=1) if len(error) > 1 else np.nan
        rows.append({**dict(zip(["regime", "model", *KEYS], key)), "logs": len(error),
                     "oracle_effect_per100": oracle, "oracle_mc_se_per100": group.oracle_mc_se_per100.iloc[0],
                     "mean_estimate_per100": group.predicted_effect_per100.mean(),
                     "bias_per100": error.mean(), "bias_se_per100": spread / np.sqrt(len(error)),
                     "sd_between_logs_per100": spread,
                     "rmse_per100": float(np.sqrt(np.mean(error ** 2))),
                     "mean_absolute_error_per100": float(np.abs(error).mean()),
                     "zero_absolute_error_per100": abs(oracle),
                     "logs_beating_zero": int(group.beats_zero.sum())})
    return per_log, pd.DataFrame(rows).sort_values([*KEYS, "regime", "model"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Directory that holds <regime>/log*/")
    parser.add_argument("--output-dir", type=Path, help="Default: <root>/summary")
    args = parser.parse_args(argv)
    output = args.output_dir or args.root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    per_log, summary = summarize(collect(args.root))
    per_log.to_csv(output / "per_log.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    shown = summary[summary.horizon == summary.horizon.max()].copy()
    shown["estimate (sd)"] = [f"{mean:+.2f} ({sd:.2f})" for mean, sd in
                              zip(shown.mean_estimate_per100, shown.sd_between_logs_per100)]
    shown["beats zero"] = [f"{wins}/{logs}" for wins, logs in zip(shown.logs_beating_zero, shown.logs)]
    shown = shown.rename(columns={"oracle_effect_per100": "oracle", "mean_absolute_error_per100": "abs error",
                                  "zero_absolute_error_per100": "zero error"})
    columns = ["policy", "outcome", "regime", "model", "oracle", "estimate (sd)", "abs error", "zero error", "beats zero"]
    print(shown[columns].round(2).to_string(index=False))
    print(f"[done] {output / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
