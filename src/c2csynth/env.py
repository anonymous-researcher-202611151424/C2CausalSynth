"""Daily closed-loop environment for sequential coupon benchmarking.

Each calendar day follows the causal order

    pre-action state -> today's coupon(s) -> today's events -> next-day state.

Real covariates initialize the population only.  Purchase, listing, sale and
coupon-exposure histories are then rolled forward one day at a time from the
simulator's own events.  This makes sequences such as item coupon -> sale ->
seller coupon three days later directly representable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from c2csynth.config import COUPON_TYPES
FACTOR_COLUMNS = []  # The item-market oracle samples private states internally.
from typing import Any
from c2csynth.initialization import sale_experience_state


SNAPSHOT_COLUMNS = [
    "user_id",
    "date",
    "group_type",
    "segment",
    "device_os",
    "region",
    "device_group",
    "days_since_signup",
    "tenure_bin",
    "region_group",
    "buy_cnt_7d",
    "buy_gmv_7d",
    "buy_cnt_30d",
    "buy_gmv_30d",
    "list_cnt_7d",
    "list_cnt_30d",
    "sold_cnt_7d_pre",
    "seller_gmv_7d_pre",
    "sold_cnt_30d_pre",
    "seller_gmv_30d_pre",
    "days_since_last_sale",
    "gap_before_last_sale",
    "active_listing_inventory",
    "buy_cnt_30d_bin",
    "buy_gmv_30d_bin",
    "list_cnt_30d_bin",
    "sold_cnt_30d_bin",
    "seller_gmv_30d_bin",
]

WARM_START_COLUMNS = [f"{key}_prior_exposures" for key in COUPON_TYPES]

ACTION_DETAIL_COLUMNS = {
    "buyer_coupon_type_f": "buyer",
    "buyer_coupon_segment_f": "buyer",
    "buyer_coupon_event_cnt": "buyer",
    "seller_coupon_event_cnt": "seller",
    "seller_event_log": "seller",
    "item_coupon_target_item_cnt": "item",
    "target_item_avg_price": "item",
    "item_price_available": "item",
    "item_cnt_log": "item",
    "item_price_log": "item",
}

ActionDetailProvider = Callable[[pd.DataFrame], pd.DataFrame]


def _count_bin(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), "4p", dtype=object)
    out[values <= 3.5] = "2_3"
    out[values <= 1.5] = "1"
    out[values <= 0.5] = "0"
    return out


def _gmv_bin(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), "3_high", dtype=object)
    out[values <= 10_000] = "2_mid"
    out[values <= 3_000] = "1_low"
    out[values <= 0.5] = "0"
    return out


def _prior_exposure_bin(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), "5p", dtype=object)
    out[values <= 4.5] = "3_4"
    out[values <= 2.5] = "2"
    out[values <= 1.5] = "1"
    out[values <= 0.5] = "0"
    return out


def _recent_exposure_bin(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), "3p", dtype=object)
    out[values <= 2.5] = "2"
    out[values <= 1.5] = "1"
    out[values <= 0.5] = "0"
    return out


def _fatigue_bin(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), "high", dtype=object)
    out[values < 1.75] = "mid"
    out[values < 0.75] = "low"
    out[values <= 0.0] = "0"
    return out


@dataclass(frozen=True)
class CouponCostConfig:
    """Transparent daily cost assumptions (not a benchmark target)."""

    buyer_cost_per_redemption: float = 250.0
    seller_cost_per_activation: float = 50.0
    item_cost_rate_on_gmv: float = 0.05


@dataclass(frozen=True)
class ListingInventoryConfig:
    """Initialization assumptions for the otherwise unobserved stock.

    The source extract has rolling list/sale counts but no active-listing
    snapshot. ``legacy_inventory_floor`` represents an older saleable listing
    that can explain a future sale when both rolling counts are zero.  After
    day zero the stock is no longer imputed: listings add units and sales
    consume units exactly.
    """

    legacy_inventory_floor: float = 1.0
    legacy_inventory_probability: float = 0.5
    legacy_units_per_recent_sale: float = 1.0


def item_coupon_eligible(state: pd.DataFrame) -> np.ndarray:
    """A product coupon requires at least one unit of pre-action stock."""
    inventory = state["active_listing_inventory"].to_numpy(dtype=float)
    if not np.isfinite(inventory).all() or (inventory < 0).any():
        raise ValueError("Item eligibility requires finite nonnegative pre-action inventory")
    return inventory >= 1


class CouponMarketEnv:
    """Vectorized one-day-step environment over a fixed user population."""

    def __init__(
        self,
        simulator: Any,
        population: pd.DataFrame,
        n_days: int = 84,
        cost_config: CouponCostConfig | None = None,
        inventory_config: ListingInventoryConfig | None = None,
        warm_start_exposures: bool = True,
        reward_mode: str = "expected",
        endogenous_history: bool = True,
        start_date: str | pd.Timestamp = "2026-01-01",
        seed: int = 0,
        action_detail_provider: ActionDetailProvider | None = None,
        item_requires_inventory: bool = True,
    ) -> None:
        if reward_mode not in ("expected", "sampled"):
            raise ValueError(reward_mode)
        if n_days < 1:
            raise ValueError("n_days must be positive")
        self.simulator = simulator
        self.n_days = n_days
        self.cost_config = cost_config or CouponCostConfig()
        self.inventory_config = inventory_config or ListingInventoryConfig()
        self.warm_start_exposures = warm_start_exposures
        self.reward_mode = reward_mode
        self.endogenous_history = endogenous_history
        self.item_requires_inventory = item_requires_inventory
        self.seed = seed
        inherited_provider = getattr(simulator, "action_detail_provider", None)
        self.action_detail_provider = action_detail_provider or inherited_provider
        if self.action_detail_provider is not None and not callable(
            self.action_detail_provider
        ):
            raise TypeError("action_detail_provider must be callable")

        snapshot = (
            population.sort_values(["user_id", "date"])
            .groupby("user_id", observed=True)
            .last()
            .reset_index()
        )
        keep_cols = [
            col
            for col in SNAPSHOT_COLUMNS + WARM_START_COLUMNS + FACTOR_COLUMNS
            if col in snapshot
        ]
        self._base_population = snapshot[keep_cols].reset_index(drop=True)
        self.population = self._base_population.copy()
        self.n_users = len(self.population)
        self.day = 0
        self.assigned_history: dict[str, np.ndarray] | None = None
        self.base_prior: dict[str, np.ndarray] | None = None
        self.activity_history: dict[str, np.ndarray] | None = None
        self.last_action_frame: pd.DataFrame | None = None
        self.active_listing_inventory: np.ndarray | None = None
        self._configured_start_date = pd.Timestamp(start_date).normalize()

    def _snapshot_values(self, col: str) -> np.ndarray:
        if col not in self.population:
            return np.zeros(self.n_users)
        return self.population[col].fillna(0).to_numpy(dtype=float)

    @staticmethod
    def _daily_history(total7: np.ndarray, total30: np.ndarray) -> np.ndarray:
        """Expand aggregate warm-start history into 30 daily bins.

        The exact pre-simulation event dates are unavailable.  We distribute
        the observed last-7-day mass uniformly over seven days and the
        remaining 30-day mass uniformly over the preceding 23 days.  From the
        first generated day onward, every bin is an actual simulated day.
        """

        recent = np.maximum(total7, 0.0) / 7.0
        older = np.maximum(total30 - total7, 0.0) / 23.0
        return np.concatenate(
            [
                np.repeat(older[:, None], 23, axis=1),
                np.repeat(recent[:, None], 7, axis=1),
            ],
            axis=1,
        )

    def reset(self) -> pd.DataFrame:
        self.day = 0
        self.population = self._base_population.copy()
        self.last_action_frame = None
        self.assigned_history = {
            key: np.zeros((self.n_users, self.n_days), dtype=float)
            for key in COUPON_TYPES
        }
        self.base_prior = {}
        for key in COUPON_TYPES:
            col = f"{key}_prior_exposures"
            self.base_prior[key] = (
                self._snapshot_values(col)
                if self.warm_start_exposures
                else np.zeros(self.n_users)
            )
        self.activity_history = {
            "buy_cnt": self._daily_history(
                self._snapshot_values("buy_cnt_7d"),
                self._snapshot_values("buy_cnt_30d"),
            ),
            "buy_gmv": self._daily_history(
                self._snapshot_values("buy_gmv_7d"),
                self._snapshot_values("buy_gmv_30d"),
            ),
            "list_cnt": self._daily_history(
                self._snapshot_values("list_cnt_7d"),
                self._snapshot_values("list_cnt_30d"),
            ),
            "sold_cnt": self._daily_history(
                self._snapshot_values("sold_cnt_7d_pre"),
                self._snapshot_values("sold_cnt_30d_pre"),
            ),
            "seller_gmv": self._daily_history(
                self._snapshot_values("seller_gmv_7d_pre"),
                self._snapshot_values("seller_gmv_30d_pre"),
            ),
        }
        if "active_listing_inventory" not in self.population:
            raise ValueError("Supply explicit initial inventory for this environment")
        inventory = self._snapshot_values("active_listing_inventory")
        self.active_listing_inventory = np.maximum(inventory, 0.0)
        self.population["active_listing_inventory"] = self.active_listing_inventory
        return self.state_frame()

    @staticmethod
    def _lag_sum(history: np.ndarray, end: int, lo: int, hi: int) -> np.ndarray:
        """Sum assignment lags lo..hi days before ``end`` (inclusive)."""

        start = max(0, end - hi)
        stop = max(0, end - lo + 1)
        if stop <= start:
            return np.zeros(history.shape[0])
        return history[:, start:stop].sum(axis=1)

    def state_frame(self) -> pd.DataFrame:
        """State immediately before the current day's action."""

        if (
            self.assigned_history is None
            or self.activity_history is None
            or self.active_listing_inventory is None
        ):
            raise RuntimeError("call reset() before state_frame()")
        frame = self.population.copy()
        current_date = self._configured_start_date + pd.Timedelta(days=self.day)
        frame["date"] = current_date
        # Same-quarter fidelity runs can reuse the fitted weekly shock.  For
        # benchmark dates outside the fit period, WeekShockModel.lookup falls
        # back to its group x segment parent multiplier as before.
        week_start = current_date - pd.Timedelta(days=int(current_date.dayofweek))
        frame["week"] = week_start.strftime("%Y-%m-%d")
        frame["organic_row"] = 0
        frame["day"] = self.day
        frame["day_of_quarter"] = self.day + 1
        frame["day_of_week"] = int(current_date.dayofweek)
        frame["is_weekend"] = int(current_date.dayofweek >= 5)
        signup = self._snapshot_values("days_since_signup")
        frame["days_since_signup"] = signup + self.day
        if "device_os" not in frame:
            frame["device_os"] = frame.get("device_group", "unknown")
        if "region" not in frame:
            frame["region"] = frame.get("region_group", "unknown")

        hist = self.activity_history
        frame["buy_cnt_7d"] = hist["buy_cnt"][:, -7:].sum(axis=1)
        frame["buy_gmv_7d"] = hist["buy_gmv"][:, -7:].sum(axis=1)
        frame["buy_cnt_30d"] = hist["buy_cnt"].sum(axis=1)
        frame["buy_gmv_30d"] = hist["buy_gmv"].sum(axis=1)
        frame["list_cnt_7d"] = hist["list_cnt"][:, -7:].sum(axis=1)
        frame["list_cnt_30d"] = hist["list_cnt"].sum(axis=1)
        frame["sold_cnt_7d_pre"] = hist["sold_cnt"][:, -7:].sum(axis=1)
        frame["seller_gmv_7d_pre"] = hist["seller_gmv"][:, -7:].sum(axis=1)
        frame["sold_cnt_30d_pre"] = hist["sold_cnt"].sum(axis=1)
        frame["seller_gmv_30d_pre"] = hist["seller_gmv"].sum(axis=1)
        frame["active_listing_inventory"] = self.active_listing_inventory
        frame["buy_cnt_30d_bin"] = _count_bin(frame["buy_cnt_30d"].to_numpy())
        frame["buy_gmv_30d_bin"] = _gmv_bin(frame["buy_gmv_30d"].to_numpy())
        frame["list_cnt_30d_bin"] = _count_bin(frame["list_cnt_30d"].to_numpy())
        frame["sold_cnt_30d_bin"] = _count_bin(frame["sold_cnt_30d_pre"].to_numpy())
        frame["seller_gmv_30d_bin"] = _gmv_bin(
            frame["seller_gmv_30d_pre"].to_numpy()
        )

        for key in COUPON_TYPES:
            history = self.assigned_history[key]
            past = history[:, : self.day]
            prior = self.base_prior[key] + past.sum(axis=1)
            exp_1_3 = self._lag_sum(history, self.day, 1, 3)
            exp_4_7 = self._lag_sum(history, self.day, 4, 7)
            exp_8_14 = self._lag_sum(history, self.day, 8, 14)
            frame[f"{key}_prior_exposures"] = prior
            frame[f"{key}_exposures_1_3"] = exp_1_3
            frame[f"{key}_exposures_4_7"] = exp_4_7
            frame[f"{key}_exposures_8_14"] = exp_8_14
            frame[f"{key}_exposures_7d"] = exp_1_3 + exp_4_7
            frame[f"{key}_exposures_14d"] = exp_1_3 + exp_4_7 + exp_8_14
            if self.day == 0:
                days_since = np.where(self.base_prior[key] > 0, 45.0, 999.0)
            else:
                marked = np.where(
                    past > 0,
                    np.arange(self.day)[None, :],
                    -1,
                )
                last = marked.max(axis=1)
                days_since = np.where(
                    last >= 0,
                    self.day - last,
                    np.where(self.base_prior[key] > 0, 45.0 + self.day, 999.0),
                )
            frame[f"{key}_days_since_exposure"] = days_since
            frame[f"{key}_prior_exposure_bin"] = _prior_exposure_bin(prior)
            frame[f"{key}_recent_exposure_bin"] = _recent_exposure_bin(
                frame[f"{key}_exposures_14d"].to_numpy(dtype=float)
            )
            frame[f"{key}_fatigue_bin"] = _fatigue_bin(prior)
        frame["any_exposures_14d"] = sum(
            frame[f"{key}_exposures_14d"] for key in COUPON_TYPES
        )

        cfg = self.simulator.response_config
        frame["sale_experience"] = sale_experience_state(
            frame,
            novelty_h=getattr(self.simulator, "novelty_h", 14.0),
            half_life_days=cfg.sale_experience_half_life_days,
            window_days=cfg.sale_experience_window_days,
        )

        frame["buyer_coupon_assigned"] = 0
        frame["seller_coupon_assigned"] = 0
        frame["item_coupon_assigned"] = 0
        frame["any_coupon_assigned"] = 0
        frame["multi_coupon_assigned"] = 0
        frame["coupon_count_today"] = 0
        frame["buyer_coupon_type_f"] = "none"
        frame["buyer_coupon_segment_f"] = "none"
        frame["buyer_campaign_selected"] = 0
        frame["buyer_control_arm"] = 0
        frame["buyer_coupon_event_cnt"] = 0.0
        frame["seller_coupon_event_cnt"] = 0.0
        frame["seller_event_log"] = 0.0
        frame["item_coupon_target_item_cnt"] = 0.0
        frame["target_item_avg_price"] = 0.0
        frame["item_price_available"] = 0
        frame["item_cnt_log"] = 0.0
        frame["item_price_log"] = 0.0
        return frame

    @staticmethod
    def _align_detail_frame(
        details: pd.DataFrame, state: pd.DataFrame, source: str
    ) -> pd.DataFrame:
        """Align optional descriptors to the environment's user order."""

        if not isinstance(details, pd.DataFrame):
            raise TypeError(f"{source} must return a pandas DataFrame")
        if "user_id" in details.columns:
            if details["user_id"].duplicated().any():
                raise ValueError(f"{source} contains duplicate user_id values")
            aligned = details.set_index("user_id").reindex(state["user_id"])
            aligned.index = state.index
            return aligned
        if len(details) != len(state):
            raise ValueError(
                f"{source} returned {len(details)} rows for {len(state)} users"
            )
        aligned = details.copy()
        aligned.index = state.index
        return aligned

    def _apply_action_details(
        self,
        frame: pd.DataFrame,
        supplied: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Resolve active-action descriptors and sanitize inactive actions.

        Descriptors come from an ``action_detail_provider`` or explicit action
        columns; explicit values take precedence. Active binary actions with
        no empirical descriptor fail fast instead of silently becoming a
        synthetic reference action.
        """

        buyer = frame["buyer_coupon_assigned"].fillna(0).to_numpy(dtype=bool)
        seller = frame["seller_coupon_assigned"].fillna(0).to_numpy(dtype=bool)
        item = frame["item_coupon_assigned"].fillna(0).to_numpy(dtype=bool)

        sources: list[tuple[pd.DataFrame, bool]] = []
        if self.action_detail_provider is not None:
            provided = self.action_detail_provider(frame.copy())
            sources.append(
                (
                    self._align_detail_frame(
                        provided, frame, "action_detail_provider"
                    ),
                    False,
                )
            )
        if supplied is not None:
            sources.append((supplied, True))

        masks = {"buyer": buyer, "seller": seller, "item": item}
        selected = frame["buyer_campaign_selected"].fillna(0).to_numpy(dtype=bool)
        declared_control = frame["buyer_control_arm"].fillna(0).to_numpy(dtype=bool)
        for detail, explicit in sources:
            for col, key in ACTION_DETAIL_COLUMNS.items():
                if col not in detail.columns:
                    continue
                values = detail[col]
                action_mask = masks[key]
                if explicit and key == "buyer":
                    action_mask = action_mask | selected | declared_control
                mask = action_mask & values.notna().to_numpy()
                frame.loc[mask, col] = values.loc[mask].to_numpy()

        # Campaign controls are policy-selected holdouts, not delivered buyer
        # coupons. They do not enter action history, cost, or response.
        selected |= declared_control
        control = selected & (
            declared_control
            | frame["buyer_coupon_type_f"].astype(str).eq("control").to_numpy()
        )
        # Backward compatibility for old replay frames that encoded campaign
        # selection as assigned=1 plus a control descriptor.
        old_control = buyer & frame["buyer_coupon_type_f"].astype(str).eq(
            "control"
        ).to_numpy()
        control |= old_control
        selected |= buyer | control
        frame.loc[control, "buyer_coupon_assigned"] = 0
        frame["buyer_campaign_selected"] = selected.astype(int)
        frame["buyer_control_arm"] = control.astype(int)
        buyer = frame["buyer_coupon_assigned"].to_numpy(dtype=bool)
        seller = frame["seller_coupon_assigned"].to_numpy(dtype=bool)
        item = frame["item_coupon_assigned"].to_numpy(dtype=bool)

        inactive_buyer_campaign = ~(buyer | control)
        frame.loc[inactive_buyer_campaign, "buyer_coupon_type_f"] = "none"
        frame.loc[inactive_buyer_campaign, "buyer_coupon_segment_f"] = "none"
        frame.loc[~buyer, "buyer_coupon_event_cnt"] = 0.0
        buyer_type = frame["buyer_coupon_type_f"].astype(str)
        buyer_segment = frame["buyer_coupon_segment_f"].astype(str)
        invalid_buyer = buyer & (
            buyer_type.isin({"none", "control"})
            | buyer_type.str.startswith("__")
            | buyer_segment.eq("none")
            | buyer_segment.str.startswith("__")
        ).to_numpy()
        if invalid_buyer.any():
            raise ValueError(
                "active buyer coupons require an empirical coupon-type and "
                "campaign-segment descriptor"
            )
        buyer_event_count = pd.to_numeric(
            frame["buyer_coupon_event_cnt"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        frame["buyer_coupon_event_cnt"] = np.where(
            buyer, np.maximum(buyer_event_count, 1.0), 0.0
        )

        seller_count = pd.to_numeric(
            frame["seller_coupon_event_cnt"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        seller_count = np.where(seller, np.maximum(seller_count, 0.0), 0.0)
        if (seller & (seller_count <= 0)).any():
            raise ValueError(
                "active seller coupons require an empirical positive event-count "
                "descriptor"
            )
        frame["seller_coupon_event_cnt"] = seller_count
        frame["seller_event_log"] = np.log1p(seller_count)

        item_count = pd.to_numeric(
            frame["item_coupon_target_item_cnt"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        item_count = np.where(item, np.maximum(item_count, 0.0), 0.0)
        if (item & (item_count <= 0)).any():
            raise ValueError(
                "active item coupons require an empirical positive target-item-count "
                "descriptor"
            )
        frame["item_coupon_target_item_cnt"] = item_count
        frame["item_cnt_log"] = np.log1p(item_count)

        price_available = pd.to_numeric(
            frame["item_price_available"], errors="coerce"
        ).fillna(0).to_numpy(dtype=bool)
        price_available &= item
        item_price = pd.to_numeric(
            frame["target_item_avg_price"], errors="coerce"
        ).to_numpy(dtype=float)
        invalid_priced = price_available & (~np.isfinite(item_price) | (item_price < 0))
        if invalid_priced.any():
            raise ValueError(
                "priced item actions require a finite nonnegative "
                "target_item_avg_price"
            )
        item_price = np.where(
            item & price_available,
            item_price,
            np.where(item, np.nan, 0.0),
        )
        frame["target_item_avg_price"] = item_price
        frame["item_price_available"] = price_available.astype(int)
        frame["item_price_log"] = np.where(
            price_available, np.log1p(np.maximum(item_price, 0.0)), 0.0
        )
        count_today = sum(
            frame[f"{key}_coupon_assigned"] for key in COUPON_TYPES
        )
        frame["coupon_count_today"] = count_today
        frame["any_coupon_assigned"] = (count_today > 0).astype(int)
        frame["multi_coupon_assigned"] = (count_today >= 2).astype(int)
        return frame

    def step(
        self, actions: pd.DataFrame, seed: int | None = None
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Apply today's actions, sample today's events, and advance one day."""

        if self.assigned_history is None:
            raise RuntimeError("call reset() before step()")
        if self.day >= self.n_days:
            raise RuntimeError("environment horizon exhausted")
        frame = self.state_frame()
        aligned = actions.set_index("user_id").reindex(frame["user_id"])
        for key in COUPON_TYPES:
            col = f"{key}_coupon_assigned"
            if col in aligned:
                assigned = aligned[col].fillna(0).to_numpy(dtype=float)
                frame[col] = assigned.astype(int)
        if self.item_requires_inventory:
            assigned = frame["item_coupon_assigned"].to_numpy() != 0
            if (assigned & ~item_coupon_eligible(frame)).any():
                raise ValueError("Item coupon assigned without pre-action inventory >= 1")
        frame["buyer_campaign_selected"] = frame["buyer_coupon_assigned"]
        for col in ["buyer_campaign_selected", "buyer_control_arm"]:
            if col in aligned:
                frame[col] = aligned[col].fillna(0).to_numpy(dtype=int)
        frame = self._apply_action_details(frame, supplied=aligned)
        for key in COUPON_TYPES:
            self.assigned_history[key][:, self.day] = frame[
                f"{key}_coupon_assigned"
            ].to_numpy(dtype=float)
        count_today = sum(
            frame[f"{key}_coupon_assigned"] for key in COUPON_TYPES
        )
        frame["coupon_count_today"] = count_today
        frame["any_coupon_assigned"] = (count_today > 0).astype(int)
        frame["multi_coupon_assigned"] = (count_today >= 2).astype(int)
        self.last_action_frame = frame.copy()

        if self.reward_mode == "expected":
            expected = self.simulator.expected_day(frame)
            outcomes = pd.DataFrame(index=frame.index)
            outcomes["purchase_cnt_1d"] = expected["expected_purchase_cnt_1d"]
            outcomes["purchase_gmv_1d"] = expected["expected_purchase_gmv_1d"]
            outcomes["listing_cnt_1d"] = expected["expected_listing_cnt_1d"]
            outcomes["sold_cnt_1d"] = expected["expected_sold_cnt_1d"]
            outcomes["seller_gmv_1d"] = expected["expected_seller_gmv_1d"]
            outcomes["bcr"] = expected["expected_bcr"]
            outcomes["lcr"] = expected["expected_lcr"]
            outcomes["scr"] = expected["expected_scr"]
            outcomes["total_gmv_1d"] = (
                outcomes["purchase_gmv_1d"] + outcomes["seller_gmv_1d"]
            )
        else:
            step_seed = self.seed + 7919 * self.day if seed is None else seed
            outcomes = self.simulator.simulate_day(frame, seed=step_seed)
        outcomes = self._apply_inventory_constraint(frame, outcomes)
        costs = self._costs(frame, outcomes)
        outcomes["coupon_cost_1d"] = costs
        outcomes["net_reward"] = (
            outcomes["purchase_gmv_1d"] + outcomes["seller_gmv_1d"] - costs
        )
        if self.endogenous_history:
            self._update_history(outcomes)

        info = {
            "day": float(self.day),
            "n_buyer_campaign_selected": float(
                frame["buyer_campaign_selected"].sum()
            ),
            "n_buyer_control": float(frame["buyer_control_arm"].sum()),
            "n_buyer_assigned": float(frame["buyer_coupon_assigned"].sum()),
            "n_seller_assigned": float(frame["seller_coupon_assigned"].sum()),
            "n_item_assigned": float(frame["item_coupon_assigned"].sum()),
            "n_multi_action": float(frame["multi_coupon_assigned"].sum()),
            "total_bcr": float(outcomes["bcr"].sum()),
            "total_lcr": float(outcomes["lcr"].sum()),
            "total_scr": float(outcomes["scr"].sum()),
        }
        self.day += 1
        return outcomes, info

    def _update_history(self, outcomes: pd.DataFrame) -> None:
        """Append today's events to exact daily rolling histories."""

        if self.activity_history is None or self.active_listing_inventory is None:
            raise RuntimeError("call reset() before stepping")
        current = {
            "buy_cnt": outcomes["purchase_cnt_1d"].to_numpy(dtype=float),
            "buy_gmv": outcomes["purchase_gmv_1d"].to_numpy(dtype=float),
            "list_cnt": outcomes["listing_cnt_1d"].to_numpy(dtype=float),
            "sold_cnt": outcomes["sold_cnt_1d"].to_numpy(dtype=float),
            "seller_gmv": outcomes["seller_gmv_1d"].to_numpy(dtype=float),
        }
        for key, values in current.items():
            history = self.activity_history[key]
            history[:, :-1] = history[:, 1:]
            history[:, -1] = values

        self.active_listing_inventory = np.maximum(
            self.active_listing_inventory
            + current["list_cnt"]
            - current["sold_cnt"],
            0.0,
        )
        self.population["active_listing_inventory"] = self.active_listing_inventory

        pop = self.population
        old_days = pop.get(
            "days_since_last_sale",
            pd.Series(np.full(self.n_users, 999.0), index=pop.index),
        ).fillna(999).to_numpy(dtype=float)
        old_gap = pop.get(
            "gap_before_last_sale",
            pd.Series(np.full(self.n_users, 30.0), index=pop.index),
        ).fillna(30).to_numpy(dtype=float)
        sale_probability = np.clip(outcomes["scr"].to_numpy(dtype=float), 0.0, 1.0)
        pop["gap_before_last_sale"] = (
            sale_probability * old_days + (1.0 - sale_probability) * old_gap
        )
        pop["days_since_last_sale"] = (
            (1.0 - sale_probability) * (old_days + 1.0)
        )

    @staticmethod
    def _apply_inventory_constraint(
        frame: pd.DataFrame, outcomes: pd.DataFrame
    ) -> pd.DataFrame:
        """Cap same-day sales by pre-action active listing inventory."""

        if "active_listing_inventory" not in frame:
            return outcomes
        out = outcomes.copy()
        inventory = np.maximum(
            frame["active_listing_inventory"].fillna(0).to_numpy(dtype=float), 0.0
        )
        raw_count = np.maximum(
            out["sold_cnt_1d"].fillna(0).to_numpy(dtype=float), 0.0
        )
        capped_count = np.minimum(raw_count, inventory)
        ratio = np.divide(
            capped_count,
            raw_count,
            out=np.zeros_like(capped_count),
            where=raw_count > 1e-12,
        )
        out["sold_cnt_1d"] = capped_count
        if "seller_gmv_1d" in out:
            out["seller_gmv_1d"] = (
                out["seller_gmv_1d"].fillna(0).to_numpy(dtype=float) * ratio
            )
        if "scr" in out:
            scr = np.clip(out["scr"].to_numpy(dtype=float), 0.0, 1.0)
            out["scr"] = scr * (inventory > 1e-12)
            if np.all(np.isin(scr, [0.0, 1.0])):
                out["scr"] = (capped_count > 0).astype(int)
        if "total_gmv_1d" in out and "purchase_gmv_1d" in out:
            out["total_gmv_1d"] = (
                out["purchase_gmv_1d"].to_numpy(dtype=float)
                + out["seller_gmv_1d"].to_numpy(dtype=float)
            )
        return out

    def _costs(self, frame: pd.DataFrame, outcomes: pd.DataFrame) -> np.ndarray:
        cfg = self.cost_config
        buyer = (
            frame["buyer_coupon_assigned"].to_numpy(dtype=float)
            * outcomes["bcr"].to_numpy(dtype=float)
            * cfg.buyer_cost_per_redemption
        )
        seller = (
            frame["seller_coupon_assigned"].to_numpy(dtype=float)
            * outcomes["lcr"].to_numpy(dtype=float)
            * cfg.seller_cost_per_activation
        )
        item = (
            frame["item_coupon_assigned"].to_numpy(dtype=float)
            * outcomes["seller_gmv_1d"].to_numpy(dtype=float)
            * cfg.item_cost_rate_on_gmv
        )
        return buyer + seller + item

    def model_uplift(
        self,
        coupon_type: str = "buyer",
        target: str | None = None,
        action_details: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """One-day conditional rate uplift under resolved action details."""

        frame = self.state_frame()
        base = self.simulator.expected_day(frame)
        treated = frame.copy()
        treated[f"{coupon_type}_coupon_assigned"] = 1
        if coupon_type == "buyer":
            treated["buyer_campaign_selected"] = 1
        treated["any_coupon_assigned"] = 1
        treated["coupon_count_today"] = 1
        aligned_details = (
            self._align_detail_frame(action_details, frame, "action_details")
            if action_details is not None
            else None
        )
        treated = self._apply_action_details(treated, supplied=aligned_details)
        lift = self.simulator.expected_day(treated)
        if target is None:
            target = {
                "buyer": "expected_bcr",
                "seller": "expected_lcr",
                "item": "expected_scr",
            }[coupon_type]
        uplift = (lift[target] - base[target]).to_numpy(dtype=float)
        if coupon_type == "item" and self.item_requires_inventory:
            uplift[~item_coupon_eligible(frame)] = 0.0
        return uplift
