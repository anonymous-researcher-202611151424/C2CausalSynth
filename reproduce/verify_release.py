#!/usr/bin/env python3
"""Check the frozen release inventory, hashes, and initial-state projection."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from c2csynth.resources import POPULATION, population_splits, world_spec

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    entries = manifest["files_sha256"]
    for relative, expected in entries.items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Missing or unsafe release path: {relative}")
        if digest(path) != expected:
            raise ValueError(f"Release file changed: {relative}")
    checksums = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        checksums[relative] = checksum
    expected_checksums = {**entries, "MANIFEST.json": digest(ROOT / "MANIFEST.json")}
    if checksums != expected_checksums:
        raise ValueError("SHA256SUMS disagrees with the release manifest")
    excluded = {".git", ".venv", ".pytest_cache", "__pycache__", "outputs", "build", "dist"}
    actual = {str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()
              and not any(part in excluded or part.endswith(".egg-info") for part in path.relative_to(ROOT).parts)
              and path.name != ".DS_Store"}
    if actual != set(entries) | {"MANIFEST.json", "SHA256SUMS"}:
        raise ValueError("Unexpected files in release; check the explicit manifest")

    spec = world_spec()
    assert spec["world_sha256"] == manifest["reference_world_sha256"]
    frame = pd.read_csv(POPULATION / "initial_states.csv.gz")
    assert len(frame) == 27000 and len(frame.columns) == 11
    assert frame.user_id.tolist() == [f"u{i:05d}" for i in range(len(frame))]
    assert frame.region.str.fullmatch(r"level_\d{3}").all()
    assert frame.device_os.str.fullmatch(r"level_\d{3}").all()
    assert not any("date" in name or "email" in name for name in frame.columns)
    with np.load(POPULATION / "splits.npz", allow_pickle=False) as data:
        assert len(data.files) == 21
        assert all(np.issubdtype(data[name].dtype, np.integer) for name in data.files)
        assert len(np.unique(np.concatenate([data[name] for name in data.files]))) == len(frame)
    test_ids = None
    for log in range(1, 11):
        populations, metadata = population_splits(log)
        assert [len(populations[name]) for name in ("train", "validation", "test")] == [10000, 5000, 3000]
        current = populations["test"].user_id.tolist()
        assert test_ids is None or test_ids == current
        test_ids = current
    return {"files": len(entries), "initial_users": len(frame), "initial_columns": len(frame.columns),
            "world_id": spec["world_id"], "status": "passed"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
