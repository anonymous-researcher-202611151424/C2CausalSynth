"""Fidelity metrics on a caller-supplied observed panel. No restricted data loader."""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from c2csynth.item_market import CHANNELS, COUPON_NAMES, CouponValidity, DailyItemMarket, DailyMarketParameters

@dataclass(frozen=True)
class MarketPanel:
    initial: pd.DataFrame
    dates: pd.DatetimeIndex
    requests: np.ndarray  # day, user, channel; item requests are counts
    buyer_arm: np.ndarray  # -1 unselected, 0 campaign holdout, 1 delivered
    observed_windows: dict[int, np.ndarray]
    inferred_single_item_requests: np.ndarray

    @property
    def observed_inventory_users(self) -> int:
        if "active_listing_inventory" not in self.initial:
            return 0
        return int(self.initial["active_listing_inventory"].notna().sum())

    def subset(self, users: np.ndarray) -> MarketPanel:
        return MarketPanel(
            self.initial.iloc[users].reset_index(drop=True), self.dates,
            self.requests[:, users], self.buyer_arm[:, users],
            {days: counts[:, users] for days, counts in self.observed_windows.items()},
            self.inferred_single_item_requests[:, users],
        )



def forward_counts(daily: np.ndarray, days: int) -> np.ndarray:
    cumulative = np.concatenate((np.zeros_like(daily[:1]), daily.cumsum(axis=0)))
    return cumulative[days:] - cumulative[:-days]



@dataclass(frozen=True)
class MarketRollout:
    counts: np.ndarray
    delivered: np.ndarray
    initial_inventory: np.ndarray
    final_inventory: np.ndarray



def replay_market(
    panel: MarketPanel, parameters: DailyMarketParameters, validity: CouponValidity, seed: int,
) -> MarketRollout:
    market = DailyItemMarket(panel.initial, parameters, validity, seed=seed, start_date=panel.dates[0])
    initial_inventory = market.inventory()
    counts = np.zeros_like(panel.requests)
    delivered = np.zeros(3, dtype=np.int64)
    for day, requests in enumerate(panel.requests):
        events = market.step(requests)
        counts[day] = events.counts
        delivered += events.delivered.sum(axis=0)
    return MarketRollout(counts, delivered, initial_inventory, market.inventory())



class Metric(NamedTuple):
    name: str
    window: int
    channel: str
    subset: str = "all"



class Measurement(NamedTuple):
    value: float
    windows: int



class MarketMoments:
    """Compare counts on identical observed user/window support in every draw."""

    def __init__(self, panel: MarketPanel) -> None:
        self.panel = panel
        self.valid = {days: np.isfinite(counts) for days, counts in panel.observed_windows.items()}
        for days, valid in self.valid.items():
            if not valid.any(axis=(0, 1)).all():
                raise ValueError(f"the user split needs uncensored {days}-day windows for each outcome")
        self.observed, self.observed_users = self.measure(panel.observed_windows)

    def measure(
        self, counts: dict[int, np.ndarray],
    ) -> tuple[dict[Metric, Measurement], dict[Metric, np.ndarray]]:
        values: dict[Metric, Measurement] = {}
        per_user: dict[Metric, np.ndarray] = {}

        def mean(name, days, channel, array, mask, subset="all"):
            size = int(mask.sum())
            if size:
                values[Metric(name, days, channel, subset)] = Measurement(
                    float(np.sum(array, where=mask) / size), size,
                )

        for days, window in counts.items():
            support = self.valid[days]
            for j, channel in enumerate(CHANNELS):
                count = window[:, :, j]
                positive = count > 0
                valid = support[:, :, j]
                windows_per_user = valid.sum(axis=0)
                covered = windows_per_user > 0
                n = windows_per_user[covered]
                if len(n) == 0:
                    continue
                incidence = (positive & valid).sum(axis=0)[covered] / n
                per_user[Metric("incidence", days, channel)] = incidence
                per_user[Metric("mean_count", days, channel)] = np.sum(count, where=valid, axis=0)[covered] / n
                mean("incidence", days, channel, positive, valid)
                mean("mean_count", days, channel, count, valid)
                mean("multiple_event_rate", days, channel, count >= 2, valid)
                values[Metric("zero_user_share", days, channel)] = Measurement(float(np.mean(incidence == 0)), int(valid.sum()))
                values[Metric("user_incidence_sd", days, channel)] = Measurement(float(np.std(incidence)), int(valid.sum()))

                # Compare like-for-like calendar blocks and windows anchored
                # at observed delivery requests. These associations are not
                # assumed to be causal intervention effects.
                for start in range(0, len(count), 7):
                    block = slice(start, start + 7)
                    mean("weekly_incidence", days, channel, positive[block], valid[block],
                         self.panel.dates[start].date().isoformat())
                requested = self.panel.requests[:len(count), :, j] > 0
                if (valid & requested).sum() >= 20:
                    mean("delivery_window_incidence", days, channel, positive, valid & requested)

            # Two nonoverlapping seven-day windows preserve the event order
            # without reconstructing event dates from rolling totals.
            if days == 7:
                for earlier, later in ((0, 0), (1, 2), (2, 1)):
                    first = window[:-7, :, earlier]
                    second = window[7:, :, later]
                    joint = support[:-7, :, earlier] & support[7:, :, later]
                    mean("lagged_joint_incidence", days, CHANNELS[later],
                         (first > 0) & (second > 0), joint, CHANNELS[earlier])

            # Standardize the recorded buyer treatment/holdout contrast to
            # dates where both arms are represented in this user split.
            treated = (self.panel.buyer_arm[:len(window)] == 1) & support[:, :, 0]
            control = (self.panel.buyer_arm[:len(window)] == 0) & support[:, :, 0]
            n_t, n_c = treated.sum(axis=1), control.sum(axis=1)
            shared = (n_t > 0) & (n_c > 0)
            if n_t[shared].sum() >= 10 and n_c[shared].sum() >= 10:
                bought = window[:, :, 0] > 0
                contrast = (bought & treated).sum(axis=1)[shared] / n_t[shared]
                contrast -= (bought & control).sum(axis=1)[shared] / n_c[shared]
                values[Metric("buyer_campaign_contrast", days, "purchase")] = Measurement(
                    float(np.average(contrast, weights=n_c[shared])), int((n_t[shared] + n_c[shared]).sum()),
                )
        return values, per_user

    def simulate(self, rollout: MarketRollout):
        return self.measure({days: forward_counts(rollout.counts, days) for days in self.valid})



@dataclass(frozen=True)
class FidelityTolerance:
    incidence_relative: float = 0.05
    incidence_absolute: float = 0.001
    count_relative: float = 0.10
    count_absolute: float = 0.002
    user_incidence_ks: float = 0.05



def fidelity_agreement(report: pd.DataFrame) -> dict:
    primary = report.loc[report["within_tolerance"].notna()]
    agreement = {}
    for name, group in [*primary.groupby("metric"), ("primary", primary)]:
        # Object columns can contain both Python and NumPy booleans. Count
        # matching rows explicitly so NumPy boolean addition cannot collapse
        # several passing rows into one True value.
        passed = int(group["within_tolerance"].eq(True).sum())
        agreement[name] = {"passed": passed, "tested": len(group), "fraction": passed / len(group)}
    return agreement



def evaluate_market(
    panel: MarketPanel, parameters: DailyMarketParameters, validity: CouponValidity,
    *, seeds: tuple[int, ...], tolerance: FidelityTolerance, n_bootstrap: int = 200,
) -> tuple[pd.DataFrame, dict]:
    moments = MarketMoments(panel)
    replications = []
    inventory = []
    deliveries = []
    requested = panel.requests.sum(axis=(0, 1))
    for seed in seeds:
        draw = replay_market(panel, parameters, validity, seed)
        replications.append(moments.simulate(draw))
        inventory.append([float(draw.initial_inventory.mean()), float(draw.final_inventory.mean()),
                          float(np.mean(draw.final_inventory == 0)), int(draw.final_inventory.max())])
        deliveries.append(draw.delivered)

    rows = []
    rng = np.random.default_rng(seeds[0])
    for key, observed in moments.observed.items():
        preds = np.array([stats[key].value for stats, _ in replications])
        prediction = float(preds.mean())
        limit = None
        ci = (np.nan, np.nan)
        if key.name in ("incidence", "mean_count"):
            rel = tolerance.incidence_relative if key.name == "incidence" else tolerance.count_relative
            abs_floor = tolerance.incidence_absolute if key.name == "incidence" else tolerance.count_absolute
            limit = max(rel * abs(observed.value), abs_floor)
            if n_bootstrap:
                users = moments.observed_users[key]
                coverage = moments.valid[key.window][:, :, CHANNELS.index(key.channel)].sum(axis=0)
                coverage = coverage[coverage > 0]
                resampled = []
                for start in range(0, n_bootstrap, 16):
                    indices = rng.integers(len(users), size=(min(16, n_bootstrap - start), len(users)))
                    weights = coverage[indices]
                    resampled.extend((users[indices] * weights).sum(axis=1) / weights.sum(axis=1))
                ci = np.quantile(resampled, (0.025, 0.975))
        error = prediction - observed.value
        rows.append({"metric": key.name, "window_days": key.window, "channel": key.channel,
                     "subset": key.subset, "observed_windows": observed.windows,
                     "observed": observed.value, "simulated": prediction, "error": error,
                     "relative_error": error / observed.value if observed.value else np.nan,
                     "observed_ci_low": ci[0], "observed_ci_high": ci[1],
                     "seed_min": preds.min(), "seed_max": preds.max(),
                     "tolerance": limit, "within_tolerance": abs(error) <= limit if limit is not None else None})
        if key.name == "incidence":
            distances = np.array([ks_2samp(moments.observed_users[key], users[key]).statistic
                                  for _, users in replications])
            rows.append({"metric": "user_incidence_ks", "window_days": key.window, "channel": key.channel,
                         "subset": key.subset, "observed_windows": observed.windows,
                         "simulated": distances.mean(), "seed_min": distances.min(), "seed_max": distances.max(),
                         "tolerance": tolerance.user_incidence_ks,
                         "within_tolerance": distances.mean() <= tolerance.user_incidence_ks})
    report = pd.DataFrame(rows)
    delivery = np.asarray(deliveries).mean(axis=0)
    stock = np.asarray(inventory)
    return report, {
        "users": len(panel.initial), "days": len(panel.dates), "seeds": list(seeds),
        "agreement": fidelity_agreement(report),
        "mean_absolute_incidence_error": float(report.loc[report.metric == "incidence", "error"].abs().mean()),
        "max_absolute_incidence_error": float(report.loc[report.metric == "incidence", "error"].abs().max()),
        "mean_absolute_count_error": float(report.loc[report.metric == "mean_count", "error"].abs().mean()),
        "max_absolute_count_error": float(report.loc[report.metric == "mean_count", "error"].abs().max()),
        "mean_user_incidence_ks": float(report.loc[report.metric == "user_incidence_ks", "simulated"].mean()),
        "max_user_incidence_ks": float(report.loc[report.metric == "user_incidence_ks", "simulated"].max()),
        "replayed_requests": {name: int(requested[i]) for i, name in enumerate(COUPON_NAMES)},
        "mean_delivered": {name: float(delivery[i]) for i, name in enumerate(COUPON_NAMES)},
        "mean_unfulfilled": {name: float(requested[i] - delivery[i]) for i, name in enumerate(COUPON_NAMES)},
        "single_item_requests_imputed": int(panel.inferred_single_item_requests.sum()),
        "inventory": {"mean_initial": float(stock[:, 0].mean()), "mean_final": float(stock[:, 1].mean()),
                      "final_zero_share": float(stock[:, 2].mean()), "max_final": int(stock[:, 3].max())},
    }
