"""Connect the item ledger to sequential policy evaluation."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pandas as pd
from c2csynth.dynamic_policy import ACTION_COLUMNS, EVENTS, RATES, responsive_target_mask
from c2csynth.env import item_coupon_eligible
from c2csynth.item_market import CouponValidity, DailyItemMarket, DailyMarketParameters
from c2csynth.initialization import draw_concentrated_inventory, _calibrated_probability

@dataclass(frozen=True)
class ItemMarketWorld:
    parameters: DailyMarketParameters
    validity: CouponValidity
    # Zero share and negative-binomial shape of the concentrated initial stock.
    inventory_zero: float = .5
    inventory_shape: float = 1.

    def initial_inventory(self, population, *, rule, seed, start_date):
        if rule == "concentrated":
            return draw_concentrated_inventory(population, self.parameters, seed, scale=1.,
                                               zero=self.inventory_zero, shape=self.inventory_shape)
        if rule == "baseline":
            return self.response(population, seed, start_date).market.inventory()
        raise ValueError(f"Unknown inventory rule: {rule}")

    def response(self, population, seed, start_date):
        return ItemMarketResponse(DailyItemMarket(
            population.sort_values("user_id"), self.parameters, self.validity, seed=seed, start_date=start_date,
        ))



class ItemMarketResponse:
    # CouponMarketEnv maintains the shared observable feature/history schema.
    response_config = SimpleNamespace(sale_experience_half_life_days=3.0,
                                      sale_experience_window_days=14)

    def __init__(self, market):
        self.market = market

    def simulate_day(self, frame, seed):
        np.testing.assert_array_equal(frame.user_id.to_numpy(dtype=str), self.market.user_ids)
        np.testing.assert_array_equal(frame.active_listing_inventory.to_numpy(), self.market.inventory())
        np.testing.assert_array_equal(frame.day.to_numpy(), self.market.day)
        requests = frame[ACTION_COLUMNS].to_numpy(dtype=int)
        draw = self.market.step(requests)
        # One item coupon per assigned user and one-day validity ensure the
        # assigned and delivered actions coincide in this benchmark.
        np.testing.assert_array_equal(draw.delivered, requests)
        outcomes = pd.DataFrame(np.column_stack((draw.counts, np.zeros((len(frame), 2)))), columns=EVENTS)
        outcomes[RATES] = (draw.counts > 0).astype(int)
        return outcomes

    def targeted_actions(self, state, gamma, probability, seed):
        """Favor the responsive target strata and optionally use private scores."""
        scores = responsive_target_mask(state, gap=3).astype(float)
        if gamma:
            scores += gamma * self.market.latent_logging_scores()
        eligible = np.ones(scores.shape, dtype=bool)
        eligible[:, 2] = item_coupon_eligible(state)
        uniforms = np.random.default_rng(seed).random(scores.shape)
        actions = np.zeros(scores.shape, dtype=int)
        for index, mask in enumerate(eligible.T):
            if mask.any():
                rate = _calibrated_probability(scores[mask, index], probability)
                actions[mask, index] = uniforms[mask, index] < rate
        out = state[["user_id"]].copy()
        out[ACTION_COLUMNS] = actions
        return out
