#!/usr/bin/env python3
"""Recompute published aggregates without observed user-level outcomes."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from c2csynth.resources import world_spec
from c2csynth.summarize import KEYS, summarize

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/paper"


def verify():
    manifest = json.loads((INPUT / "manifest.json").read_text())
    assert manifest["oracle_matches_current_parameters"] is True
    spec = world_spec()
    if manifest.get("world_sha256") != spec["world_sha256"]:
        raise ValueError("Paper aggregates identify another reference world")
    per_log = pd.read_csv(INPUT / "benchmark_per_log.csv")
    # Each released row is already averaged over learning seeds on one log.
    # Reapply the reported grouping and error calculation across those logs.
    _, recomputed = summarize(per_log.assign(run=per_log.model))
    expected = pd.read_csv(INPUT / "benchmark_summary.csv")
    order = ["regime", "model", *KEYS]
    assert_frame_equal(recomputed.sort_values(order).reset_index(drop=True),
                       expected.sort_values(order).reset_index(drop=True),
                       check_dtype=False, check_exact=False, rtol=1e-11, atol=1e-12)

    with np.load(INPUT / "oracle_aggregate.npz", allow_pickle=False) as data:
        names = data["policies"].tolist() if "policies" in data else data["policy_names"].tolist()
        events = data["events"]
        assert events.shape == (manifest["oracle_mc"], 3, 42, 3)
        cumulative = events.cumsum(axis=2) * 100
        difference = cumulative[:, :2] - cumulative[:, [2]]
        means = difference.mean(axis=0)
        errors = difference.std(axis=0, ddof=1) / np.sqrt(len(difference))
    assert names == ["logging_025", "responsive_025", "noop"]
    for row in expected.itertuples():
        i = names.index(row.policy)
        j = ("bcr", "lcr", "scr").index(row.outcome)
        assert row.reference_policy == "noop"
        np.testing.assert_allclose([row.oracle_effect_per100, row.oracle_mc_se_per100],
            [means[i, row.horizon - 1, j], errors[i, row.horizon - 1, j]], rtol=1e-11, atol=1e-12)

    summaries = json.loads((INPUT / "summary.json").read_text())
    for name, summary in summaries.items():
        frame = pd.read_csv(INPUT / f"{name}_fidelity.csv")
        recomputed = {
            "max_abs_relative_incidence_error": frame.loc[frame.metric == "incidence", "relative_error"].abs().max(),
            "max_abs_relative_count_error": frame.loc[frame.metric == "mean_count", "relative_error"].abs().max(),
            "max_user_incidence_ks": frame.loc[frame.metric == "user_incidence_ks", "simulated"].max(),
        }
        assert set(summary) == set(recomputed)
        for key, value in summary.items():
            np.testing.assert_allclose(value, recomputed[key], rtol=1e-12, atol=1e-14)
    return {"benchmark_summary_rows": len(expected), "benchmark_log_rows": len(per_log),
            "oracle_replicates": len(events), "fidelity_conditions": len(summaries),
            "world_id": spec["world_id"], "status": "passed"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
