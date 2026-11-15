"""Shared I/O and factual-loss helpers."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from c2csynth.dynamic_policy import one_hot

def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()



def load_log(path):
    return pd.read_csv(path, dtype={"user_id": str}, keep_default_na=False)



def verify_files(directory, recorded, names=None):
    for name in recorded if names is None else names:
        if digest(Path(directory) / name) != recorded[name]:
            raise ValueError(f"Artifact has changed since generation: {name}")



def tensor_batch(arrays, rows, device):
    values = {key: torch.as_tensor(value[rows], device=device) for key, value in arrays.items()}
    # The generic GT's action input carries observed pre-action context plus
    # action features. It is an input token only; it is never predicted.
    return [one_hot(values["previous_classes"]), values["previous_events"],
            values["context"], values["classes"], values["events"]]



def fitted_loss(model, arrays, batch_size, device):
    totals = np.zeros(3)
    n = len(arrays["context"])
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = slice(start, min(start + batch_size, n))
            loss = model.training_loss(*tensor_batch(arrays, rows, device))
            count = min(batch_size, n - start)
            totals += count * np.array([loss.total.item(), loss.categorical_cross_entropy.item(),
                                       loss.continuous_mse.item()])
    return dict(zip(["total", "categorical_ce", "continuous_mse"], (totals / n).tolist()))
