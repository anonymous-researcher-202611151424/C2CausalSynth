"""Short-horizon policy evaluation with learned daily event transitions.

Only whitelisted observed features enter estimators. Generated events update
inventory, rolling histories and exposure histories through the public
CouponMarketEnv bookkeeping. No fitted structural response is used by learners.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from c2csynth.config import COUPON_TYPES
from c2csynth.env import CouponMarketEnv, SNAPSHOT_COLUMNS, WARM_START_COLUMNS, item_coupon_eligible

RATES = ["bcr", "lcr", "scr"]
EVENTS = ["purchase_cnt_1d", "listing_cnt_1d", "sold_cnt_1d",
          "purchase_gmv_1d", "seller_gmv_1d"]
STATE_COLUMNS = [
    "buy_cnt_7d", "buy_cnt_30d", "buy_gmv_7d", "buy_gmv_30d",
    "list_cnt_7d", "list_cnt_30d", "sold_cnt_7d_pre", "sold_cnt_30d_pre",
    "seller_gmv_7d_pre", "seller_gmv_30d_pre", "active_listing_inventory",
    "days_since_last_sale", "gap_before_last_sale", "days_since_signup",
    "day", "day_of_week",
] + [f"{action}_{suffix}" for action in COUPON_TYPES for suffix in
     ["prior_exposures", "exposures_1_3", "exposures_4_7", "exposures_8_14",
      "days_since_exposure"]]
ACTION_COLUMNS = [f"{a}_coupon_assigned" for a in COUPON_TYPES]
# Bookkeeping of earlier assignments; the encoder context starts with STATE_COLUMNS.
EXPOSURE_HISTORY_INDICES = [i for i, name in enumerate(STATE_COLUMNS) if "exposure" in name]
DETAIL_COLUMNS = ["buyer_coupon_event_cnt", "seller_coupon_event_cnt",
                  "item_coupon_target_item_cnt", "target_item_avg_price",
                  "item_price_available"]
CATEGORY_COLUMNS = ["device_os", "region", "segment",
                    "buyer_coupon_type_f", "buyer_coupon_segment_f"]
POLICIES = ("responsive_gap1", "responsive_gap3")
POLICY_COMPARISONS = {
    "gap1-vs-gap3": POLICIES,
    "dynamic-vs-noop": ("responsive_gap1", "noop"),
    "logging-vs-noop": ("logging_025", "noop"),
}
CLEAR_POLICIES = ("logging_025", "responsive_025", "noop")
SALE_INDEX = RATES.index("scr")
# Public feasibility bit, appended last to the estimator context and left
# unstandardized: a sale (like an item coupon) needs pre-action inventory >= 1.
DERIVED_COLUMNS = ["sale_feasible"]




@dataclass(frozen=True)
class FixedDescriptors:
    """One supported descriptor configuration, identical across all branches."""
    values: dict[str, Any]

    def __post_init__(self):
        values = {key: value.item() if isinstance(value, np.generic) else value
                  for key, value in self.values.items()}
        available = values.get("item_price_available")
        if available not in (0, 1):
            raise ValueError("item_price_available must be zero or one")
        if available == 0:
            # Missing price denotes a real coupon program, not a free item.
            # JSON null preserves this meaning; the env keeps the raw NaN.
            values["target_item_avg_price"] = None
        else:
            price = values.get("target_item_avg_price")
            if price is None or not np.isfinite(float(price)) or float(price) < 0:
                raise ValueError("Available item price must be finite and nonnegative")
        for key, value in values.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"Non-finite descriptor: {key}")
        object.__setattr__(self, "values", values)

    def __call__(self, state: pd.DataFrame) -> pd.DataFrame:
        values = dict(self.values)
        if values["target_item_avg_price"] is None:
            # Environment frames use a numerical missing value, not an
            # object column of None (which also triggers pandas dtype warnings).
            values["target_item_avg_price"] = np.nan
        return pd.DataFrame({key: [value] * len(state)
                             for key, value in values.items()}, index=state.index)




def random_logging_actions(state, rng, probability=0.025, *, item_requires_inventory=True):
    """Independent daily/channel allocation, shared by logging and evaluation."""
    if not 0 < probability < 1:
        raise ValueError("Logging probability must be strictly between zero and one")
    out = state[["user_id"]].copy()
    for col in ACTION_COLUMNS:
        out[col] = (rng.random(len(state)) < probability).astype(int)
    if item_requires_inventory:
        out.loc[~item_coupon_eligible(state), "item_coupon_assigned"] = 0
    return out


def responsive_target_mask(state: pd.DataFrame, gap: int) -> np.ndarray:
    eligible = np.column_stack([
        state["buy_cnt_7d"].to_numpy() < 1,
        state["days_since_last_sale"].to_numpy() <= 3,
        item_coupon_eligible(state),
    ])
    exposure_gaps = state[[f"{key}_days_since_exposure" for key in COUPON_TYPES]].to_numpy()
    return eligible & (exposure_gaps >= gap)


def policy_actions(state: pd.DataFrame, policy: str, *, rng=None,
                   item_requires_inventory=True) -> pd.DataFrame:
    """Observable adaptive policies; no oracle effects or hidden factors."""
    if policy in ("logging_025", "responsive_025"):
        if rng is None:
            raise ValueError("Stochastic policy requires an explicit RNG")
        out = random_logging_actions(state, rng, item_requires_inventory=item_requires_inventory)
        if policy == "responsive_025":
            out[ACTION_COLUMNS] = out[ACTION_COLUMNS].to_numpy() * responsive_target_mask(state, gap=3)
        return out
    if policy not in (*POLICIES, "noop"):
        raise ValueError(f"Unknown policy: {policy}")
    out = state[["user_id"]].copy()
    if policy == "noop":
        out[ACTION_COLUMNS] = 0
        return out
    gap = 1 if policy == "responsive_gap1" else 3
    out[ACTION_COLUMNS] = responsive_target_mask(state, gap).astype(int)
    return out


def observable_initial(state: pd.DataFrame) -> pd.DataFrame:
    """Persist the initial inventory once, then remove latent/oracle fields."""
    required = [c for c in dict.fromkeys(SNAPSHOT_COLUMNS + WARM_START_COLUMNS)
                if c in state]
    out = state[required].copy()
    if "active_listing_inventory" not in out:
        raise ValueError("Resolve initial inventory with the oracle before stripping Z")
    out["group_type"] = "UTG"
    return out


def make_env(simulator, population, descriptors, days, seed, start_date, *, item_requires_inventory=True):
    population = population.copy()
    population["group_type"] = "UTG"
    return CouponMarketEnv(simulator, population, n_days=days, seed=seed,
                           start_date=start_date, reward_mode="sampled",
                           endogenous_history=True, action_detail_provider=descriptors,
                           item_requires_inventory=item_requires_inventory)


def generate_random_log(simulator, population, descriptors, days, seed, start_date,
                        action_probability=0.25, *, item_requires_inventory=True,
                        logging="randomized", gamma=0.0, targeted_actions=None, never_assigned=None):
    """Factual log under randomized or state-targeted assignment.

    ``logging="targeted"`` draws each channel from the fitted observable-state
    propensity score shifted by ``gamma`` times the user's latent engagement and
    recentred to the same mean rate. ``gamma=0`` is targeting on observables;
    ``gamma>0`` is hidden confounding, because the logged panel omits Z.
    Users in ``never_assigned`` are a holdout arm: no coupon under either regime.
    """
    if not 0 < action_probability < 1:
        raise ValueError("Random logging probability must be strictly between zero and one")
    if logging not in ("randomized", "targeted") or gamma < 0 or (logging == "randomized" and gamma):
        raise ValueError("Unknown logging regime")
    env = make_env(simulator, population, descriptors, days, seed, start_date,
                   item_requires_inventory=item_requires_inventory)
    env.reset()
    rng = np.random.default_rng(seed + 17011)
    chunks = []
    for day in range(days):
        state = env.state_frame()
        if logging == "randomized":
            actions = random_logging_actions(state, rng, action_probability,
                                             item_requires_inventory=item_requires_inventory)
        elif targeted_actions is not None:
            actions = targeted_actions(state, gamma, action_probability, seed + 17011 + 7919 * day)
        else:
            raise ValueError("Targeted logging requires the world targeting callback")
        if never_assigned is not None:
            actions.loc[actions["user_id"].isin(never_assigned).to_numpy(), ACTION_COLUMNS] = 0
        outcomes, _ = env.step(actions, seed=seed + 7919 * day)
        # Explicit projection prevents future targets, Z and oracle columns
        # from entering the model input through an automatically inferred schema.
        observed = env.last_action_frame
        columns = ["user_id"] + STATE_COLUMNS + ACTION_COLUMNS + DETAIL_COLUMNS + CATEGORY_COLUMNS
        logged = observed[columns].copy()
        for col in EVENTS + RATES:
            logged[col] = outcomes[col].to_numpy()
        chunks.append(logged)
    return pd.concat(chunks, ignore_index=True)


@dataclass
class FeatureEncoder:
    numeric_mean: np.ndarray | None = None
    numeric_scale: np.ndarray | None = None
    event_mean: np.ndarray | None = None
    event_scale: np.ndarray | None = None
    category_maps: dict[str, list[str]] = field(default_factory=dict)

    @staticmethod
    def raw_numeric(frame):
        numeric = frame[STATE_COLUMNS + ACTION_COLUMNS + DETAIL_COLUMNS].copy()
        available = pd.to_numeric(numeric["item_price_available"], errors="raise").to_numpy(float)
        if not np.isin(available, [0, 1]).all():
            raise ValueError("item_price_available must be zero or one")
        price = pd.to_numeric(numeric["target_item_avg_price"], errors="coerce").to_numpy(float)
        if ((available == 1) & (~np.isfinite(price) | (price < 0))).any():
            raise ValueError("Available item price must be finite and nonnegative")
        # Only the model's numerical input gets a placeholder. The separate
        # availability bit distinguishes missing from an observed zero price.
        numeric["target_item_avg_price"] = np.where(available == 1, price, 0.0)
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Observed numerical inputs must be finite and nonnegative")
        return np.log1p(values)

    def fit(self, train):
        values = self.raw_numeric(train)
        self.numeric_mean = values.mean(axis=0)
        self.numeric_scale = np.maximum(values.std(axis=0), 0.1)
        events = np.log1p(train[EVENTS].to_numpy(dtype=float))
        self.event_mean = events.mean(axis=0)
        self.event_scale = np.maximum(events.std(axis=0), 0.1)
        self.category_maps = {col: sorted(train[col].fillna("unknown").astype(str).unique())
                              for col in CATEGORY_COLUMNS}
        return self

    def context(self, frame):
        values = (self.raw_numeric(frame) - self.numeric_mean) / self.numeric_scale
        blocks = [values]
        for col, vocab in self.category_maps.items():
            lookup = {value: i + 1 for i, value in enumerate(vocab)}
            codes = frame[col].fillna("unknown").astype(str).map(lookup).fillna(0).to_numpy(int)
            blocks.append(np.eye(len(vocab) + 1)[codes])
        # The inventory rule is public bookkeeping, so learners receive the same
        # feasibility bit that the environment applies to sales and item coupons.
        blocks.append(item_coupon_eligible(frame).astype(float)[:, None])
        return np.concatenate(blocks, axis=-1).astype(np.float32)

    def events(self, events):
        return ((np.log1p(events) - self.event_mean) / self.event_scale).astype(np.float32)

    def inverse_events(self, normalized):
        logged = np.asarray(normalized) * self.event_scale + self.event_mean
        if not np.isfinite(logged).all() or (logged > 30).any():
            raise FloatingPointError("Unstable event rollout; inspect model before increasing horizon")
        return np.expm1(np.maximum(logged, 0))

    def to_dict(self):
        return {"numeric_mean": self.numeric_mean.tolist(),
                "numeric_scale": self.numeric_scale.tolist(),
                "event_mean": self.event_mean.tolist(), "event_scale": self.event_scale.tolist(),
                "category_maps": self.category_maps, "state_columns": STATE_COLUMNS,
                "action_columns": ACTION_COLUMNS, "detail_columns": DETAIL_COLUMNS,
                "event_columns": EVENTS, "derived_columns": DERIVED_COLUMNS}

    @classmethod
    def from_dict(cls, data):
        if (data["state_columns"] != STATE_COLUMNS or data["event_columns"] != EVENTS
                or data.get("derived_columns") != DERIVED_COLUMNS):
            raise ValueError("Feature contract mismatch")
        return cls(**{key: np.asarray(data[key]) for key in
                      ["numeric_mean", "numeric_scale", "event_mean", "event_scale"]},
                   category_maps=data["category_maps"])


def sequence_arrays(frame, encoder):
    frame = frame.sort_values(["user_id", "day"]).reset_index(drop=True)
    groups = frame.groupby("user_id", sort=False)
    lengths = groups.size().to_numpy()
    if not len(lengths) or np.unique(lengths).size != 1:
        raise ValueError("A balanced user-day panel is required")
    n, days = len(lengths), int(lengths[0])
    day = frame["day"].to_numpy().reshape(n, days)
    if not np.all(day == np.arange(days)):
        raise ValueError("Days must start at zero and be contiguous, without duplicates")
    raw_events = frame[EVENTS].to_numpy(float).reshape(n, days, len(EVENTS))
    classes = frame[RATES].to_numpy(np.int64).reshape(n, days, 3)
    if not np.isin(classes, [0, 1]).all():
        raise ValueError("Training requires sampled binary events, not expected-mode outcomes")
    previous_classes = np.concatenate([np.zeros((n, 1, 3), dtype=int), classes[:, :-1]], axis=1)
    previous_events = np.concatenate([np.zeros((n, 1, 5)), raw_events[:, :-1]], axis=1)
    return {
        "context": encoder.context(frame).reshape(n, days, -1),
        "previous_classes": previous_classes,
        "previous_events": encoder.events(previous_events),
        "classes": classes, "events": encoder.events(raw_events),
    }


def assignment_feature_mask(encoder):
    """Context columns known before today's assignment (no actions or descriptors)."""
    numeric = STATE_COLUMNS + ACTION_COLUMNS + DETAIL_COLUMNS
    keep = [name in STATE_COLUMNS for name in numeric]
    for col in CATEGORY_COLUMNS:
        keep += [not col.startswith("buyer_coupon_")] * (len(encoder.category_maps[col]) + 1)
    return np.array(keep + [True] * len(DERIVED_COLUMNS))


def ipw_policy_rates(frame, encoder, policies, *, probability=0.025, clip=(1e-3, .5), truncate=.99):
    """Per-decision, self-normalized inverse-probability weighting on a factual log.

    Assignment probabilities are estimated from the observed pre-assignment state
    by one logistic regression per channel; the true logging law is never read.
    Returns daily event probabilities [policy, day, outcome] of the logged cohort
    under each target policy, with effective sample sizes at the final day.
    """
    frame = frame.sort_values(["user_id", "day"]).reset_index(drop=True)
    days = int(frame["day"].nunique())
    users = len(frame) // days
    x = encoder.context(frame)[:, assignment_feature_mask(encoder)]
    actions = frame[ACTION_COLUMNS].to_numpy(int)
    feasible = np.ones(actions.shape, dtype=bool)
    feasible[:, ACTION_COLUMNS.index("item_coupon_assigned")] = item_coupon_eligible(frame)
    if (actions[~feasible] != 0).any():
        raise ValueError("Factual log contains an infeasible assignment")
    propensity = np.zeros(actions.shape)
    for k in range(actions.shape[1]):
        rows = feasible[:, k]
        if not rows.any():
            continue
        target = actions[rows, k]
        fitted = (np.full(rows.sum(), target.mean()) if len(np.unique(target)) < 2 else
                  LogisticRegression(C=1.0, max_iter=400).fit(x[rows], target).predict_proba(x[rows])[:, 1])
        propensity[rows, k] = np.clip(fitted, *clip)
    gates = responsive_target_mask(frame, gap=3)
    events = frame[RATES].to_numpy(float).reshape(users, days, len(RATES))
    rates, effective, unsupported = [], {}, {}
    for policy in policies:
        if policy == "noop":
            law = np.zeros(actions.shape)
        elif policy == "logging_025":
            law = probability * feasible
        elif policy == "responsive_025":
            law = probability * gates
        else:
            raise ValueError(f"Unknown policy: {policy}")
        ratio = np.where(actions == 1, law / np.maximum(propensity, clip[0]), (1 - law) / (1 - propensity))
        ratio[~feasible] = 1.0  # both laws assign zero probability to an infeasible action
        weight = np.cumprod(ratio.prod(axis=1).reshape(users, days), axis=1)
        weight = np.minimum(weight, np.quantile(weight, truncate, axis=0))
        total = weight.sum(axis=0)
        # A day on which no logged trajectory is compatible with the target policy has no
        # weighted estimate; it keeps the unweighted logged rate and is counted below.
        supported = total > 0
        weighted = (weight[:, :, None] * events).sum(axis=0) / np.where(supported, total, 1.)[:, None]
        rates.append(np.where(supported[:, None], weighted, events.mean(axis=0)))
        unsupported[policy] = int((~supported).sum())
        final = weight[:, -1]
        effective[policy] = float(final.sum() ** 2 / np.square(final).sum()) if final.any() else 0.
    return np.stack(rates), dict(users=users, effective_users_final_day=effective,
                                 unsupported_days=unsupported,
                                 propensity_clip=list(clip), weight_truncation_quantile=truncate)


def one_hot(classes):
    if isinstance(classes, torch.Tensor):
        return torch.nn.functional.one_hot(classes.long(), 2).flatten(-2).float()
    return np.eye(2, dtype=np.float32)[classes].reshape(*classes.shape[:-1], 6)


class MarkovTransition:
    """Simple g-computation baseline using current summaries, without a Transformer.

    ``kind="logistic"`` is linear in the encoded state; ``kind="gbt"`` replaces the
    three occurrence models with gradient-boosted trees and keeps the ridge model
    for event quantities. ``exposure_history=False`` removes the bookkeeping of
    earlier assignments from the predictors: a coupon then acts on its own day,
    and on later days only through the events it generated.
    """
    def __init__(self, kind="logistic", exposure_history=True):
        if kind not in ("logistic", "gbt"):
            raise ValueError(f"Unknown Markov transition kind: {kind}")
        self.kind = kind
        self.exposure_history = exposure_history

    def _inputs(self, context, previous_events, previous_classes):
        if not getattr(self, "exposure_history", True):
            context = np.delete(context, EXPOSURE_HISTORY_INDICES, axis=-1)
        return np.concatenate([context, previous_events, one_hot(previous_classes)], axis=-1)

    def _classifier(self):
        if getattr(self, "kind", "logistic") == "gbt":
            return HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15,
                                                  min_samples_leaf=200, l2_regularization=1.0,
                                                  early_stopping=False, random_state=0)
        return LogisticRegression(C=1.0, max_iter=400)

    def fit(self, train, validation):
        def inputs(data):
            x = self._inputs(data["context"], data["previous_events"], data["previous_classes"])
            return x.reshape(-1, x.shape[-1])
        x = inputs(train)
        target = train["classes"].reshape(-1, 3)
        # Sale occurrence is modeled on feasible user-days only; an infeasible
        # day has probability zero by the public inventory rule.
        feasible = train["context"][..., -1].reshape(-1) > .5
        if target[~feasible, SALE_INDEX].any():
            raise ValueError("Training log contains a sale without pre-action inventory")
        self.classifiers = []
        for k in range(3):
            rows = feasible if k == SALE_INDEX else np.ones(len(x), dtype=bool)
            if not rows.any():
                self.classifiers.append(0.)
            elif len(np.unique(target[rows, k])) < 2:
                self.classifiers.append(float(target[rows, k].mean()))
            else:
                self.classifiers.append(self._classifier().fit(x[rows], target[rows, k]))
        self.continuous = Ridge(alpha=10.0).fit(np.c_[x, target], train["events"].reshape(-1, 5))
        val_x = inputs(validation)
        predicted = self.continuous.predict(np.c_[val_x, validation["classes"].reshape(-1, 3)])
        self.residuals = validation["events"].reshape(-1, 5) - predicted
        return self

    def draw(self, context, previous_events, previous_classes, seed):
        x = self._inputs(context, previous_events, previous_classes)
        rng = np.random.default_rng(seed)
        probabilities = np.stack([np.full(len(x), clf) if isinstance(clf, float)
                                  else clf.predict_proba(x)[:, 1] for clf in self.classifiers], axis=-1)
        probabilities[:, SALE_INDEX] *= context[:, -1] > .5
        classes = (rng.random(probabilities.shape) < probabilities).astype(int)
        means = self.continuous.predict(np.c_[x, classes])
        # Match the ordinary GT's marginal empirical-residual sampler.
        indices = rng.integers(len(self.residuals), size=means.shape)
        residuals = self.residuals[indices, np.arange(5)[None, :]]
        return classes, means + residuals


def project_events(classes, values, inventory):
    """Public support constraints; no probabilities or fitted response parameters."""
    values = np.maximum(np.asarray(values).copy(), 0)
    values[:, :3] = np.maximum(np.rint(values[:, :3]), 1) * classes
    values[:, 3] *= classes[:, 0]
    values[:, 4] *= classes[:, 2]
    # Only pre-action inventory is saleable. New same-day listings cannot sell.
    raw_sales = values[:, 2].copy()
    values[:, 2] = np.minimum(values[:, 2], np.floor(np.maximum(inventory, 0)))
    values[:, 4] *= np.divide(values[:, 2], raw_sales, out=np.zeros_like(raw_sales), where=raw_sales > 0)
    out = pd.DataFrame(values, columns=EVENTS)
    for i, name in enumerate(RATES):
        out[name] = (values[:, i] > 0).astype(float)
    out["total_gmv_1d"] = values[:, 3] + values[:, 4]
    return out


class LearnedResponse:
    """Implements only event generation; CouponMarketEnv handles known state updates."""
    response_config = SimpleNamespace(sale_experience_half_life_days=3.0,
                                      sale_experience_window_days=14)

    def __init__(self, model, encoder, *, residuals=None, device="cpu", batch_size=128, count_only=False):
        self.model, self.encoder, self.residuals = model, encoder, residuals
        self.device, self.batch_size = device, batch_size
        self.count_only = count_only
        self.context_history, self.class_history, self.event_history = [], [], []
        self.previous_classes = self.previous_events = None

    @torch.no_grad()
    def simulate_day(self, frame, seed):
        n = len(frame)
        if self.previous_classes is None:
            self.previous_classes = np.zeros((n, 3), dtype=int)
            self.previous_events = np.zeros((n, 5))
        context = self.encoder.context(frame)
        events = self.encoder.events(self.previous_events)
        if isinstance(self.model, MarkovTransition):
            classes, continuous = self.model.draw(context, events, self.previous_classes, seed)
        else:
            self.context_history.append(context)
            self.class_history.append(self.previous_classes.copy())
            self.event_history.append(events)
            contexts = np.stack(self.context_history, axis=1)
            categories = one_hot(np.stack(self.class_history, axis=1))
            history = np.stack(self.event_history, axis=1)
            draws = []
            generator = torch.Generator(device="cpu").manual_seed(seed)
            for start in range(0, n, self.batch_size):
                stop = start + self.batch_size
                inputs = [torch.as_tensor(arr[start:stop], dtype=torch.float32, device=self.device)
                          for arr in [categories, history, contexts]]
                draw = self.model.sample_next(*inputs, self.residuals, generator=generator)
                draws.append((draw.categorical_classes.cpu().numpy(), draw.continuous.cpu().numpy()))
            classes = np.concatenate([d[0] for d in draws])
            continuous = np.concatenate([d[1] for d in draws])
        values = self.encoder.inverse_events(continuous)
        if self.count_only:
            values[:, 3:] = 0
        out = project_events(classes, values, frame["active_listing_inventory"].to_numpy())
        # Feedback uses constrained events, not unprojected samples.
        self.previous_classes = out[RATES].to_numpy(int)
        self.previous_events = out[EVENTS].to_numpy(float)
        return out


def rollout_policy(simulator, population, descriptors, policy, horizon, seed, start_date,
                   *, policy_seed=None, descriptor_seed=None, item_requires_inventory=True):
    env = make_env(simulator, population, descriptors, horizon, seed, start_date,
                   item_requires_inventory=item_requires_inventory)
    env.reset()
    action_rng = np.random.default_rng((seed if policy_seed is None else policy_seed) + 17011)
    detail_seed = seed if descriptor_seed is None else descriptor_seed
    values, actions = [], []
    for day in range(horizon):
        state = env.state_frame()
        intervention = policy_actions(state, policy, rng=action_rng,
                                      item_requires_inventory=item_requires_inventory)
        outcome, _ = env.step(intervention, seed=seed + 7919 * day)
        values.append(outcome[RATES].to_numpy(float))
        actions.append(intervention[ACTION_COLUMNS].to_numpy(float))
    return np.stack(values, axis=1), np.stack(actions, axis=1)


def mc_se(values):
    """Conditional Monte Carlo SE across independent trajectory replicates."""
    if len(values) < 2:
        raise ValueError("At least two independent MC replicates are required")
    return np.std(values, axis=0, ddof=1) / np.sqrt(len(values))


def score_policy_samples(samples, truth, action_samples, truth_actions, policies=POLICIES):
    """All metrics are population averages, never latent-conditioned individual PEHE.

    Arrays have [MC, policy, user, day, channel]. MC replicates within each
    method are paired across policies; oracle and model streams are independent.
    """
    if samples.shape[1:] != truth.shape[1:]:
        raise ValueError("Oracle and model must evaluate identical policies/users/horizons")
    if len(policies) != 2 or len(set(policies)) != 2 or samples.shape[1] != 2:
        raise ValueError("Exactly two distinct ordered policies are required")
    if action_samples.shape != samples.shape or truth_actions.shape != truth.shape:
        raise ValueError("Action and event sample dimensions must match")
    model_daily = samples.mean(axis=2)
    oracle_daily = truth.mean(axis=2)
    result = {"unit": "expected event-days per user (cumulative); probability (daily)",
              "uncertainty": "Monte Carlo only, conditional on fixed world/users/fitted model",
              "policies": {}, "contrasts": {}}
    for i, name in enumerate(policies):
        result["policies"][name] = {}
        for k, rate in enumerate(RATES):
            predicted = model_daily[:, i, :, k]
            actual = oracle_daily[:, i, :, k]
            result["policies"][name][rate] = {
                "daily_predicted": predicted.mean(axis=0).tolist(),
                "daily_oracle": actual.mean(axis=0).tolist(),
                "daily_oracle_mc_se": mc_se(actual).tolist(),
                "daily_predicted_mc_se": mc_se(predicted).tolist(),
                "daily_population_rmse": float(np.sqrt(np.mean((predicted.mean(0) - actual.mean(0)) ** 2))),
                "cumulative_predicted": float(predicted.sum(1).mean()),
                "cumulative_oracle": float(actual.sum(1).mean()),
                "cumulative_error": float(predicted.sum(1).mean() - actual.sum(1).mean()),
            }
    for k, rate in enumerate(RATES):
        pred = model_daily[:, 0, :, k] - model_daily[:, 1, :, k]
        actual = oracle_daily[:, 0, :, k] - oracle_daily[:, 1, :, k]
        pred_cum, truth_cum = pred.sum(1), actual.sum(1)
        result["contrasts"][rate] = {
            "direction": f"{policies[0]} minus {policies[1]}",
            "daily_effect_predicted": pred.mean(0).tolist(),
            "daily_effect_oracle": actual.mean(0).tolist(),
            "daily_effect_rmse": float(np.sqrt(np.mean((pred.mean(0) - actual.mean(0)) ** 2))),
            "cumulative_effect_predicted": float(pred_cum.mean()),
            "cumulative_effect_oracle": float(truth_cum.mean()),
            "cumulative_effect_error": float(pred_cum.mean() - truth_cum.mean()),
            "cumulative_effect_absolute_error": float(abs(pred_cum.mean() - truth_cum.mean())),
            "oracle_effect_mc_se": float(mc_se(truth_cum)),
            "predicted_effect_mc_se": float(mc_se(pred_cum)),
            "error_mc_se": float(np.hypot(mc_se(truth_cum), mc_se(pred_cum))),
            "zero_effect_absolute_error": float(abs(truth_cum.mean())),
        }
    result["mean_coupons_per_user"] = {
        name: {"predicted": action_samples[:, i].sum(axis=2).mean(axis=(0, 1)).tolist(),
               "oracle": truth_actions[:, i].sum(axis=2).mean(axis=(0, 1)).tolist()}
        for i, name in enumerate(policies)}
    return result
