"""Run a policy with the same reference world used in the benchmark."""
import argparse
import json

import numpy as np

from c2csynth import DailyItemMarket, DailyMarketParameters, CouponValidity
from c2csynth.resources import world_spec, population_splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--users", type=int, default=256)
    p.add_argument("--days", type=int, default=21)
    p.add_argument("--seed", type=int, default=2027)
    a = p.parse_args()
    if a.days < 1:
        p.error("days must be positive")
    populations, meta = population_splits(1, sizes={"train": 1, "validation": 1, "test": a.users})
    spec = world_spec()
    users = populations["test"].sort_values("user_id")
    market = DailyItemMarket(users, DailyMarketParameters(**spec["parameters"]),
        CouponValidity(**spec["initialization"]["coupon_valid_days"]), seed=a.seed,
        start_date=meta["calendar_origin"])
    rng = np.random.default_rng(a.seed + 17011)
    totals = np.zeros(3)
    offers = np.zeros(3)
    for _ in range(a.days):
        observed = market.observed_users()
        requests = (rng.random((len(users), 3)) < .025).astype(int)
        requests[:, 2] *= observed.active_listing_inventory.to_numpy() >= 1
        events = market.step(requests)
        totals += (events.counts > 0).sum(axis=0)
        offers += events.delivered.sum(axis=0)
    print(json.dumps({"world": spec["world_id"], "users": len(users), "days": a.days,
        "outcome_order": ["purchase", "listing", "sale"],
        "event_days_per_100": (totals * 100 / len(users)).tolist(),
        "delivered_assignments_per_100": (offers * 100 / len(users)).tolist(),
        "purpose": "API usage example; use the benchmark command for paper results"}, indent=2))


if __name__ == "__main__":
    main()
