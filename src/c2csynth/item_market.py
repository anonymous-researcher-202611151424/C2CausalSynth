"""Daily user actions and sales of individual listings.

The user, item, and coupon tables are columnar arrays during a rollout.
Realized events update engagement, satiation, and inventory. Engagement
returns to the user's baseline at a rate specific to the buying and the
listing side. A sale raises listing engagement, more so after a long gap
since the previous sale. A delivered buyer coupon lowers listing
engagement and a delivered lister coupon lowers purchase engagement.
Coupon exposure builds fatigue that recovers with the time since the last
delivery, and a coupon's lift grows with the user's private propensity on
its own side.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, fields

import numpy as np
import pandas as pd
from scipy.special import expit


CHANNELS = ("purchase", "listing", "sale")
COUPON_NAMES = ("buyer coupon", "lister coupon", "item coupon")
FRESH_ITEM_DAYS = 7
# Bounds are structural assumptions, in daily log-odds units where applicable.
PARAMETER_BOUNDS = {
    "purchase_intercept": (-10.0, -1.5),
    "listing_intercept": (-10.0, -1.5),
    "sale_intercept": (-11.0, -1.5),
    "purchase_user_sd": (0.0, 1.8),
    "listing_user_sd": (0.0, 1.8),
    "sale_user_sd": (0.0, 1.8),
    "purchase_extra_count": (0.0, 3.0),
    "listing_extra_count": (0.0, 5.0),
    "initial_extra_inventory": (0.05, 6.0),
    "purchase_engagement_retention": (0.86, 0.985),
    "listing_engagement_retention": (0.70, 0.985),
    "engagement_noise": (0.01, 0.35),
    "weekend_shift": (-0.6, 0.6),
    "purchase_satiation": (0.2, 2.0),
    "satiation_retention": (0.2, 0.85),
    "purchase_engagement_share": (0.0, 0.5),
    "sale_to_listing_gain": (0.01, 2.0),
    "buyer_coupon_lift": (0.01, 1.5),
    "listing_coupon_lift": (0.01, 1.5),
    "item_coupon_lift": (0.01, 1.5),
    "fatigue_increment": (0.0, 1.0),
    "fatigue_retention": (0.2, 0.95),
    # Engagement lost on the other side at the first delivery of a coupon.
    "buyer_coupon_listing_drag": (0.0, 0.5),
    "listing_coupon_purchase_drag": (0.0, 0.5),
    # Days since the previous sale at which a sale gives half of the gain;
    # zero gives every sale the full gain.
    "sale_gap_saturation_days": (0.0, 30.0),
    "fresh_item_sale_lift": (0.0, 1.5),
    # Fatigue per log prior delivery, halved after this many days without one.
    "cumulative_fatigue": (0.0, 1.0),
    "fatigue_recovery_days": (7.0, 30.0),
    # Log lift per unit of private propensity on the coupon's own side.
    "latent_response_slope": (0.0, 0.5),
    "latent_correlation": (0.0, 0.95),
}


@dataclass(frozen=True)
class DailyMarketParameters:
    purchase_intercept: float = -4.8
    listing_intercept: float = -5.4
    sale_intercept: float = -5.6
    purchase_user_sd: float = 0.7
    listing_user_sd: float = 0.8
    sale_user_sd: float = 0.6
    purchase_extra_count: float = 0.15
    listing_extra_count: float = 0.25
    initial_extra_inventory: float = 1.0
    purchase_engagement_retention: float = 0.92
    listing_engagement_retention: float = 0.92
    engagement_noise: float = 0.08
    weekend_shift: float = 0.0
    purchase_satiation: float = 1.5
    satiation_retention: float = 0.65
    purchase_engagement_share: float = 0.15
    # The sale experience starts above every coupon lift.
    sale_to_listing_gain: float = 1.0
    buyer_coupon_lift: float = 0.4
    listing_coupon_lift: float = 0.5
    item_coupon_lift: float = 0.5
    fatigue_increment: float = 0.35
    fatigue_retention: float = 0.85
    buyer_coupon_listing_drag: float = 0.0
    listing_coupon_purchase_drag: float = 0.0
    sale_gap_saturation_days: float = 0.0
    fresh_item_sale_lift: float = 0.0
    cumulative_fatigue: float = 0.0
    fatigue_recovery_days: float = 14.0
    latent_response_slope: float = 0.0
    latent_correlation: float = 0.0
    # Parameter files written before the retention was split set both sides.
    engagement_retention: InitVar[float | None] = None

    def __post_init__(self, engagement_retention: float | None) -> None:
        if engagement_retention is not None:
            object.__setattr__(self, "purchase_engagement_retention", engagement_retention)
            object.__setattr__(self, "listing_engagement_retention", engagement_retention)
        for field in fields(self):
            low, high = PARAMETER_BOUNDS[field.name]
            value = getattr(self, field.name)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{field.name} must be in [{low}, {high}]")


@dataclass(frozen=True)
class CouponValidity:
    """Number of eligible days including the day of delivery."""

    buyer: int = 1
    listing: int = 1
    item: int = 1

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{field.name} validity must be a positive integer")


class _ItemTable:
    def __init__(self, initial_owners: np.ndarray) -> None:
        self.owner = initial_owners.astype(np.int64, copy=True)
        self.listed_day = np.full(len(self.owner), -1, dtype=np.int64)
        self.sold_day = np.full(len(self.owner), -1, dtype=np.int64)
        self.coupon_id = np.full(len(self.owner), -1, dtype=np.int64)

    def add(self, owners: np.ndarray, day: int) -> np.ndarray:
        new_ids = np.arange(len(self.owner), len(self.owner) + len(owners))
        if len(new_ids):
            self.owner = np.concatenate((self.owner, owners))
            self.listed_day = np.concatenate((self.listed_day, np.full(len(owners), day)))
            self.sold_day = np.concatenate((self.sold_day, np.full(len(owners), -1)))
            self.coupon_id = np.concatenate((self.coupon_id, np.full(len(owners), -1)))
        return new_ids

    def available(self, day: int) -> np.ndarray:
        return np.flatnonzero((self.listed_day < day) & (self.sold_day < 0))


class _CouponTable:
    def __init__(self) -> None:
        self.owner = np.empty(0, dtype=np.int64)
        self.channel = np.empty(0, dtype=np.int8)
        self.item_id = np.empty(0, dtype=np.int64)
        self.issued_day = np.empty(0, dtype=np.int64)
        self.expiry_day = np.empty(0, dtype=np.int64)
        self.redeemed_day = np.empty(0, dtype=np.int64)
        self.lift = np.empty(0, dtype=float)

    def issue(self, owners, channel, item_ids, day, validity, lift) -> np.ndarray:
        size = len(owners)
        ids = np.arange(len(self.owner), len(self.owner) + size)
        if size:
            self.owner = np.concatenate((self.owner, owners))
            self.channel = np.concatenate((self.channel, np.full(size, channel, dtype=np.int8)))
            self.item_id = np.concatenate((self.item_id, item_ids))
            self.issued_day = np.concatenate((self.issued_day, np.full(size, day)))
            self.expiry_day = np.concatenate((self.expiry_day, np.full(size, day + validity)))
            self.redeemed_day = np.concatenate((self.redeemed_day, np.full(size, -1)))
            self.lift = np.concatenate((self.lift, lift))
        return ids

    def active(self, ids: np.ndarray, day: int) -> np.ndarray:
        result = ids.copy()
        present = ids >= 0
        known_ids = ids[present]
        result[present] = np.where(
            (self.expiry_day[known_ids] > day) & (self.redeemed_day[known_ids] < 0),
            known_ids, -1,
        )
        return result

    def active_lifts(self, active_ids: np.ndarray) -> np.ndarray:
        result = np.zeros(active_ids.shape)
        active = active_ids >= 0
        result[active] = self.lift[active_ids[active]]
        return result

    def redeem(self, ids: np.ndarray, day: int) -> None:
        self.redeemed_day[ids[ids >= 0]] = day


@dataclass(frozen=True)
class DayExpectations:
    event_probability: np.ndarray
    mean_count: np.ndarray
    eligible_item_ids: np.ndarray
    item_sale_probability: np.ndarray


@dataclass(frozen=True)
class DayEvents:
    counts: np.ndarray
    delivered: np.ndarray  # Buyer coupon, lister coupon, and item coupon counts.
    created_item_ids: np.ndarray
    sold_item_ids: np.ndarray
    targeted_item_ids: np.ndarray


@dataclass(frozen=True)
class _DayPlan:
    user_event_probability: np.ndarray
    eligible_item_ids: np.ndarray
    item_sale_probability: np.ndarray
    active_user_coupons: np.ndarray
    active_item_coupons: np.ndarray
    new_user_offers: np.ndarray
    new_item_offer_positions: np.ndarray
    new_user_lifts: np.ndarray
    new_item_lifts: np.ndarray
    engagement: np.ndarray  # After today's deliveries, before today's events.


class DailyItemMarket:
    """Users buy and list, and each available item draws its own sale event.

    Requests have one row per initial user, in that order. Columns are buyer
    coupon (0/1), lister coupon (0/1), and item coupon (a count).
    Items listed today enter tomorrow's inventory. The external side of each
    transaction is implicit, so purchases are not matched to simulated sellers.
    """

    def __init__(
        self, users: pd.DataFrame, parameters: DailyMarketParameters | None = None,
        validity: CouponValidity | None = None, *, seed: int = 0,
        start_date: str | pd.Timestamp = "2026-01-01",
    ) -> None:
        if "user_id" not in users or users.empty or users["user_id"].isna().any():
            raise ValueError("users must have nonmissing user_id values")
        self.user_ids = users["user_id"].astype(str).to_numpy(copy=True)
        if len(np.unique(self.user_ids)) != len(self.user_ids):
            raise ValueError("user_id must be unique")
        self.n_users = len(self.user_ids)
        self.parameters = parameters or DailyMarketParameters()
        self.validity = validity or CouponValidity()
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.seed = seed
        self.start_date = pd.Timestamp(start_date).normalize()
        if pd.isna(self.start_date):
            raise ValueError("start_date must be a valid date")
        self.day = 0
        p = self.parameters

        past_7 = self._history(users, ("buy_cnt_7d", "list_cnt_7d", "sold_cnt_7d_pre"))
        past_30 = self._history(users, ("buy_cnt_30d", "list_cnt_30d", "sold_cnt_30d_pre"))
        rng = self._rng(0)
        user_sd = np.array((p.purchase_user_sd, p.listing_user_sd, p.sale_user_sd))
        offsets = 0.4 * (np.log1p(np.minimum(past_30, 30)) - np.log(2))
        draws = np.clip(rng.normal(size=(self.n_users, 3)), -2.5, 2.5)
        # One shared draw correlates a user's propensities to buy, list and sell.
        shared = np.clip(self._rng(5).normal(size=(self.n_users, 1)), -2.5, 2.5)
        draws = np.sqrt(1 - p.latent_correlation) * draws + np.sqrt(p.latent_correlation) * shared
        self._latent_user_offsets = user_sd * draws
        self._user_logits = (
            np.array((p.purchase_intercept, p.listing_intercept, p.sale_intercept))
            + offsets + self._latent_user_offsets
        )
        self.engagement = np.minimum(0.5 * np.log1p(past_7[:, :2]), 1.5)
        self.satiation = (
            p.purchase_satiation * (1 - p.satiation_retention ** 7)
            * np.minimum(past_7[:, 0] / 7, 1)
        )
        recent_unsold = np.maximum(past_30[:, 1] - past_30[:, 2], 0).astype(np.int64)
        stock = recent_unsold + rng.poisson(p.initial_extra_inventory + 0.5 * np.sqrt(past_30[:, 2]))
        if "active_listing_inventory" in users:
            inventory = users["active_listing_inventory"].to_numpy(dtype=float, na_value=np.nan)
            observed = ~np.isnan(inventory)
            known = inventory[observed]
            if not np.isfinite(known).all() or np.any(known < 0) or np.any(known != np.floor(known)):
                raise ValueError("observed initial inventory must be a nonnegative integer")
            stock[observed] = known.astype(np.int64)
        self._items = _ItemTable(np.repeat(np.arange(self.n_users), stock))
        self._coupons = _CouponTable()
        self._user_coupons = np.full((self.n_users, 2), -1, dtype=np.int64)
        self._fatigue = np.zeros((self.n_users, 3))
        self._last_offer_day = np.zeros((self.n_users, 3))
        # The input has trailing counts but no sale dates, so a recent sale
        # is placed at the middle of the window that contains it.
        if "days_since_last_sale" in users:
            since_sale = users["days_since_last_sale"].fillna(999).to_numpy(dtype=float)
            if not np.isfinite(since_sale).all() or np.any(since_sale < 0):
                raise ValueError("initial days_since_last_sale must be finite and nonnegative")
        else:
            since_sale = np.select([past_7[:, 2] > 0, past_30[:, 2] > 0], [3.0, 18.0], 999.0)
        self._last_sale_day = -since_sale
        self._initial_7 = past_7
        self._initial_30 = past_30
        self._count_history = np.zeros((30, self.n_users, 3), dtype=np.int64)
        self._offer_history = np.zeros((14, self.n_users, 3), dtype=bool)
        self._prior_offers = np.zeros((self.n_users, 3), dtype=np.int64)

    @staticmethod
    def _history(users: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
        history = users.reindex(columns=columns).fillna(0).to_numpy(dtype=float)
        if not np.isfinite(history).all() or np.any(history < 0):
            raise ValueError(f"initial {columns} must be finite and nonnegative")
        return history

    def _rng(self, stream: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.day, stream]))

    def _owners_count(self, owners: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        return np.bincount(owners, weights=weights, minlength=self.n_users)

    def inventory(self) -> np.ndarray:
        return self._owners_count(self._items.owner[self._items.available(self.day)])

    def latent_logging_scores(self) -> np.ndarray:
        """Private logit contributions for logging with unobserved targeting."""
        return self._latent_user_offsets + np.column_stack((
            self.engagement[:, 0] - self.satiation,
            self.engagement[:, 1],
            0.35 * self.engagement[:, 1],
        ))

    def observed_users(self) -> pd.DataFrame:
        """Policy-facing snapshot with initial summaries and generated histories."""
        observed = pd.DataFrame({"user_id": self.user_ids, "day": self.day,
                                 "date": self.start_date + pd.Timedelta(days=self.day),
                                 "active_listing_inventory": self.inventory()})
        recent = self._count_history[(self.day - np.arange(1, 8)) % 30].sum(axis=0)
        monthly = self._count_history.sum(axis=0)
        recent_offers = self._offer_history.sum(axis=0)
        for j, name in enumerate(CHANNELS):
            observed[f"initial_{name}_count_7d"] = self._initial_7[:, j]
            observed[f"initial_{name}_count_30d"] = self._initial_30[:, j]
            observed[f"{name}_generated_count_7d"] = recent[:, j]
            observed[f"{name}_generated_count_30d"] = monthly[:, j]
            observed[f"{name}_offers_14d"] = recent_offers[:, j]
            observed[f"{name}_prior_offers"] = self._prior_offers[:, j]
        return observed

    def items(self) -> pd.DataFrame:
        items = self._items
        return pd.DataFrame({"item_id": np.arange(len(items.owner)),
                             "user_id": self.user_ids[items.owner],
                             "listed_day": items.listed_day.copy(),
                             "sold_day": items.sold_day.copy(),
                             "available": (items.listed_day < self.day) & (items.sold_day < 0)})

    def coupons(self) -> pd.DataFrame:
        coupons = self._coupons
        return pd.DataFrame({"coupon_id": np.arange(len(coupons.owner)),
                             "user_id": self.user_ids[coupons.owner],
                             "channel": np.array(["buyer", "listing", "item"])[coupons.channel],
                             "coupon_name": np.asarray(COUPON_NAMES)[coupons.channel],
                             "item_id": coupons.item_id.copy(),
                             "issued_day": coupons.issued_day.copy(),
                             "expiry_day_exclusive": coupons.expiry_day.copy(),
                             "redeemed_day": coupons.redeemed_day.copy()})

    def _choose_item_targets(self, owners, active_coupons, targets) -> np.ndarray:
        candidates = np.flatnonzero((active_coupons < 0) & (targets[owners] > 0))
        if len(candidates) == 0:
            return candidates
        candidate_owners = owners[candidates]
        order = np.lexsort((self._rng(1).random(len(candidates)), candidate_owners))
        ordered_owners = candidate_owners[order]
        positions = np.arange(len(order))
        group_starts = np.r_[True, ordered_owners[1:] != ordered_owners[:-1]]
        ranks = positions - np.maximum.accumulate(np.where(group_starts, positions, 0))
        return candidates[order[ranks < targets[ordered_owners]]]

    def _plan(self, requests: np.ndarray | None) -> _DayPlan:
        if requests is None:
            requests = np.zeros((self.n_users, 3), dtype=np.int64)
        else:
            requests = np.asarray(requests)
        if (requests.shape != (self.n_users, 3) or not np.isfinite(requests).all()
                or np.any(requests < 0) or np.any(requests != np.floor(requests))
                or np.any(requests[:, :2] > 1)):
            raise ValueError("requests must be N x 3 with binary user offers and integer item counts")

        items = self._items.available(self.day)
        owners = self._items.owner[items]
        active_users = self._coupons.active(self._user_coupons, self.day)
        active_items = self._coupons.active(self._items.coupon_id[items], self.day)
        user_offers = (requests[:, :2] > 0) & (active_users < 0)
        item_offer_positions = self._choose_item_targets(owners, active_items, requests[:, 2])

        p = self.parameters
        # Besides the fast-decaying state, fatigue grows with the log number
        # of prior deliveries and recovers with the days since the last one.
        since_offer = self.day - self._last_offer_day
        fatigue = self._fatigue + (p.cumulative_fatigue * np.log1p(self._prior_offers)
                                   * p.fatigue_recovery_days / (since_offer + p.fatigue_recovery_days))
        # A delivered coupon moves attention away from the other side: a
        # buyer coupon lowers listing engagement and a lister coupon lowers
        # purchase engagement, with the same fatigue attenuation as the lift.
        # Item coupons leave the owner's engagement unchanged.
        attenuation = user_offers / (1 + fatigue[:, :2])
        engagement = np.clip(self.engagement - np.column_stack((
            p.listing_coupon_purchase_drag * attenuation[:, 1],
            p.buyer_coupon_listing_drag * attenuation[:, 0],
        )), -2.0, 2.0)
        # Lift grows with the private propensity on the coupon's own side, so
        # it is larger after a sale and smaller after a coupon for the other
        # side, including one delivered on the same day.
        propensity = self._latent_user_offsets + np.column_stack(
            (engagement[:, 0], engagement[:, 1], 0.35 * engagement[:, 1]))
        lift = np.array((p.buyer_coupon_lift, p.listing_coupon_lift, p.item_coupon_lift))
        attenuated_lifts = lift * np.exp(p.latent_response_slope * propensity) / (1 + fatigue)
        new_user_lifts = attenuated_lifts[:, :2] * user_offers
        new_item_lifts = attenuated_lifts[owners[item_offer_positions], 2]
        user_lifts = self._coupons.active_lifts(active_users) + new_user_lifts
        item_lifts = self._coupons.active_lifts(active_items)
        item_lifts[item_offer_positions] = new_item_lifts

        weekend = (self.start_date.dayofweek + self.day) % 7 >= 5
        calendar = p.weekend_shift * weekend
        user_logits = self._user_logits[:, :2] + engagement + user_lifts + calendar
        user_logits[:, 0] -= self.satiation
        listed = self._items.listed_day[items]
        fresh = (listed >= 0) & (self.day - listed <= FRESH_ITEM_DAYS)
        item_logits = (self._user_logits[owners, 2] + 0.35 * engagement[owners, 1] + item_lifts + calendar
                       + p.fresh_item_sale_lift * fresh)
        return _DayPlan(expit(user_logits), items, expit(item_logits), active_users, active_items,
                        user_offers, item_offer_positions, new_user_lifts, new_item_lifts, engagement)

    def expected_day(self, requests: np.ndarray | None = None) -> DayExpectations:
        """Conditional expectations from the distributions used by step()."""
        plan = self._plan(requests)
        owner = self._items.owner[plan.eligible_item_ids]
        sale_count = self._owners_count(owner, plan.item_sale_probability)
        log_no_sale = self._owners_count(owner, np.log1p(-plan.item_sale_probability))
        p = self.parameters
        means = plan.user_event_probability * (1 + np.array((p.purchase_extra_count, p.listing_extra_count)))
        return DayExpectations(
            np.column_stack((plan.user_event_probability, -np.expm1(log_no_sale))),
            np.column_stack((means, sale_count)), plan.eligible_item_ids, plan.item_sale_probability,
        )

    def step(self, requests: np.ndarray | None = None) -> DayEvents:
        plan = self._plan(requests)
        p = self.parameters
        rng = self._rng(2)
        event = rng.random((self.n_users, 2)) < plan.user_event_probability
        extra = rng.poisson((p.purchase_extra_count, p.listing_extra_count), size=(self.n_users, 2))
        user_counts = event * (1 + extra)
        # Keep the same draw for an item when other items become unavailable.
        item_draws = self._rng(3).random(len(self._items.owner))
        item_sold = item_draws[plan.eligible_item_ids] < plan.item_sale_probability
        sold_ids = plan.eligible_item_ids[item_sold]
        sold_counts = self._owners_count(self._items.owner[sold_ids])
        counts = np.column_stack((user_counts, sold_counts))
        targeted_ids = plan.eligible_item_ids[plan.new_item_offer_positions]
        target_owners = self._items.owner[targeted_ids]
        delivered = np.column_stack((plan.new_user_offers, self._owners_count(target_owners)))

        for channel, validity in enumerate((self.validity.buyer, self.validity.listing)):
            owners = np.flatnonzero(plan.new_user_offers[:, channel])
            ids = self._coupons.issue(owners, channel, np.full(len(owners), -1), self.day,
                                      validity, plan.new_user_lifts[owners, channel])
            self._user_coupons[owners, channel] = ids
            plan.active_user_coupons[owners, channel] = ids
        item_coupon_ids = self._coupons.issue(
            target_owners, 2, targeted_ids, self.day, self.validity.item, plan.new_item_lifts,
        )
        self._items.coupon_id[targeted_ids] = item_coupon_ids
        plan.active_item_coupons[plan.new_item_offer_positions] = item_coupon_ids
        self._coupons.redeem(plan.active_user_coupons[event], self.day)
        self._coupons.redeem(plan.active_item_coupons[item_sold], self.day)

        self._items.sold_day[sold_ids] = self.day
        new_items = self._items.add(np.repeat(np.arange(self.n_users), counts[:, 1]), self.day)
        self._count_history[self.day % 30] = counts
        self._offer_history[self.day % 14] = delivered > 0
        self._prior_offers += delivered > 0
        self._last_offer_day[delivered > 0] = self.day
        self._fatigue = p.fatigue_retention * self._fatigue + p.fatigue_increment * (delivered > 0)

        satiation_increment = (1 - p.satiation_retention) * p.purchase_satiation * event[:, 0]
        self.satiation = p.satiation_retention * self.satiation + satiation_increment
        # A sale after a long gap is a stronger experience than one in a run of sales.
        sold = sold_counts > 0
        gap = np.maximum(self.day - self._last_sale_day, 1.0)
        novelty = gap / (gap + p.sale_gap_saturation_days)
        self._last_sale_day[sold] = self.day
        gains = np.column_stack((p.purchase_engagement_share * satiation_increment,
                                 p.sale_to_listing_gain * novelty * sold))
        noise = self._rng(4).normal(0, p.engagement_noise, size=self.engagement.shape)
        retention = np.array((p.purchase_engagement_retention, p.listing_engagement_retention))
        self.engagement = np.clip(retention * plan.engagement + gains + noise, -2.0, 2.0)
        self.day += 1
        return DayEvents(counts, delivered, new_items, sold_ids, targeted_ids)
