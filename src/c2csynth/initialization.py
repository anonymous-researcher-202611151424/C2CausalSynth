"""Deterministic initialization and targeting helpers."""
import numpy as np
import pandas as pd

def sale_experience_state(
    df: pd.DataFrame,
    novelty_h: float = 14.0,
    half_life_days: float = 3.0,
    window_days: int = 14,
) -> np.ndarray:
    """Gap-weighted, continuously decaying post-sale experience state."""

    if "sale_experience" in df:
        return np.clip(
            df["sale_experience"].fillna(0).to_numpy(dtype=float), 0.0, 1.0
        )
    days = df.get(
        "days_since_last_sale", pd.Series(999.0, index=df.index)
    ).fillna(999).to_numpy(dtype=float)
    gap = df.get(
        "gap_before_last_sale", pd.Series(30.0, index=df.index)
    ).fillna(30).to_numpy(dtype=float)
    novelty_h = max(float(novelty_h), 1e-6)
    half_life_days = max(float(half_life_days), 1e-6)
    novelty = np.maximum(gap, 0.0) / (np.maximum(gap, 0.0) + novelty_h)
    active = (days >= 0) & (days <= window_days)
    decay = np.zeros(len(df))
    decay[active] = 2.0 ** (-days[active] / half_life_days)
    return np.clip(novelty * decay, 0.0, 1.0)



def _calibrated_probability(logit: np.ndarray, rate: float) -> np.ndarray:
    """Shift logits so their mean probability equals ``rate``."""

    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        p = 1.0 / (1.0 + np.exp(-np.clip(logit + mid, -30, 30)))
        if float(p.mean()) > rate:
            hi = mid
        else:
            lo = mid
    return 1.0 / (1.0 + np.exp(-np.clip(logit + (lo + hi) / 2.0, -30, 30)))



def draw_concentrated_inventory(users, parameters, seed, *, scale, zero, shape):
    """Draw zero-inflated negative binomial stock from baseline histories."""
    listed = users["list_cnt_30d"].fillna(0).to_numpy(dtype=float)
    sold = users["sold_cnt_30d_pre"].fillna(0).to_numpy(dtype=float)
    mean = np.maximum(listed - sold, 0) + parameters.initial_extra_inventory + 0.5 * np.sqrt(sold)
    mean = scale * mean / (1 - zero)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 710]))
    keep = rng.random(len(users)) >= zero
    return rng.negative_binomial(shape, shape / (shape + mean)) * keep
