"""Load the released, frozen world and initial-state contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
REFERENCE_WORLD = RESOURCE_ROOT / "worlds/reference"
POPULATION = RESOURCE_ROOT / "population"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_resource_files(directory, record):
    directory = Path(directory)
    for name, expected in record.items():
        path = directory / name
        if path.resolve().parent != directory.resolve() or not path.is_file():
            raise ValueError(f"Invalid resource file: {name}")
        if sha256(path) != expected:
            raise ValueError(f"Frozen resource has changed: {name}")


def world_spec(directory=REFERENCE_WORLD):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    verify_resource_files(directory, manifest["files_sha256"])
    return {
        "parameters": json.loads((directory / "parameters.json").read_text()),
        "inventory_rule": json.loads((directory / "inventory_rule.json").read_text()),
        "initialization": json.loads((directory / "initialization.json").read_text()),
        "world_id": manifest["world_id"],
        "world_sha256": sha256(directory / "manifest.json"),
    }


def population_splits(log=1, *, sizes=None, directory=POPULATION):
    """Return fixed, disjoint populations in the original selection order.

    The public IDs preserve the sort order used by the simulation. Subsetting
    the released inventory does not redraw it or change its allocation seed.
    """
    if not 1 <= log <= 10:
        raise ValueError("Released log numbers are 1 through 10")
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    verify_resource_files(directory, metadata["files_sha256"])
    frame = pd.read_csv(directory / "initial_states.csv.gz", dtype={"user_id": str})
    if frame.columns.tolist() != metadata["columns"] or len(frame) != metadata["rows"]:
        raise ValueError("Initial population schema or row count changed")
    if frame.user_id.duplicated().any() or not frame.user_id.is_monotonic_increasing:
        raise ValueError("Initial population IDs must be unique and ordered")
    frame["date"] = pd.Timestamp(metadata["calendar_origin"])
    frame["segment"] = metadata["fixed_segment"]
    frame["group_type"] = metadata["fixed_group"]
    frame["days_since_last_sale"] = np.select(
        [frame.sold_cnt_7d_pre.fillna(0) > 0, frame.sold_cnt_30d_pre.fillna(0) > 0],
        [3, 18], default=999,
    )
    frame["gap_before_last_sale"] = 30
    keys = {"train": f"train_{log:02d}", "validation": f"validation_{log:02d}", "test": "test"}
    result = {}
    with np.load(directory / "splits.npz", allow_pickle=False) as splits:
        for name, key in keys.items():
            indices = splits[key]
            count = len(indices) if sizes is None else sizes[name]
            if not 0 < count <= len(indices):
                raise ValueError(f"Requested {name} size outside released split")
            result[name] = frame.iloc[indices[:count]].copy()
    ids = [set(f.user_id) for f in result.values()]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError("Population splits overlap")
    return result, metadata
