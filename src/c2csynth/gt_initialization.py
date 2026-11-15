"""Factual initialization of the Compact GT baseline."""
from __future__ import annotations
import numpy as np
import torch
from c2csynth.dynamic_policy import ACTION_COLUMNS, CATEGORY_COLUMNS, DETAIL_COLUMNS, EVENTS, RATES, STATE_COLUMNS

def treatment_context_indices(encoder, exposure_history=True):
    """Indices in the encoder context, including every treatment one-hot bit.

    ``exposure_history=False`` keeps today's assignment and its descriptors only.
    """
    numeric = STATE_COLUMNS + ACTION_COLUMNS + DETAIL_COLUMNS
    exposure = [f"{action}_{suffix}" for action in ("buyer", "seller", "item")
                for suffix in ("prior_exposures", "exposures_1_3", "exposures_4_7",
                               "exposures_8_14", "days_since_exposure")]
    selected = (exposure if exposure_history else []) + ACTION_COLUMNS + DETAIL_COLUMNS
    indices = [numeric.index(c) for c in selected]
    offset = len(numeric)
    for col in CATEGORY_COLUMNS:
        width = len(encoder.category_maps[col]) + 1
        if col.startswith("buyer_coupon_"):
            indices.extend(range(offset, offset + width))
            selected.append(col)
        offset += width
    return indices, selected



@torch.no_grad()
def initialize_factual_heads(model, train_frame, encoder):
    """Shared factual initialization; current no-action rows may have past treatment."""
    control = train_frame.loc[train_frame[ACTION_COLUMNS].eq(0).all(axis=1)]
    if control.empty:
        raise ValueError("Need factual no-action rows for initialization")
    positives = control[RATES].sum().to_numpy(float)
    probability = (positives + .5) / (len(control) + 1)
    for head, p in zip(model.categorical_heads, probability, strict=True):
        head.weight.zero_()
        head.bias.copy_(torch.tensor([np.log1p(-p), np.log(p)],
                                    dtype=head.bias.dtype, device=head.bias.device))
    continuous = encoder.events(control[EVENTS].to_numpy(float)).mean(axis=0)
    model.continuous_head.weight.zero_()
    model.continuous_head.bias.copy_(torch.as_tensor(
        continuous, dtype=model.continuous_head.bias.dtype, device=model.continuous_head.bias.device))
    return {"no_action_rows": len(control), "positive_events": positives.tolist(),
            "initial_probabilities": probability.tolist(), "probability_smoothing": "(positives + 0.5)/(n + 1)",
            "initial_normalized_continuous_means": continuous.tolist(),
            "source": "factual training rows with all current actions zero; prior exposure allowed"}
