import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from c2csynth.resources import REFERENCE_WORLD, POPULATION, world_spec, population_splits
from c2csynth.item_market import CouponValidity, DailyMarketParameters


def test_changed_world_is_rejected(tmp_path):
    world = tmp_path / "world"
    shutil.copytree(REFERENCE_WORLD, world)
    path = world / "parameters.json"
    parameters = json.loads(path.read_text())
    parameters["purchase_intercept"] += .01
    path.write_text(json.dumps(parameters))
    with pytest.raises(ValueError, match="Frozen resource has changed"):
        world_spec(world)


def test_changed_population_is_rejected(tmp_path):
    population = tmp_path / "population"
    shutil.copytree(POPULATION, population)
    path = population / "initial_states.csv.gz"
    frame = pd.read_csv(path)
    frame.loc[0, "active_listing_inventory"] += 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Frozen resource has changed"):
        population_splits(directory=population)


def test_all_log_splits_preserve_fixed_inventory_and_hold_out_evaluation_users():
    initial = pd.read_csv(POPULATION / "initial_states.csv.gz").set_index("user_id")
    common = None
    for log in range(1, 11):
        populations, _ = population_splits(log)
        train, validation, test = [set(populations[name].user_id) for name in ("train", "validation", "test")]
        assert not (train & validation or train & test or validation & test)
        assert common is None or common == test
        common = test
        for frame in populations.values():
            np.testing.assert_array_equal(frame.active_listing_inventory, initial.loc[frame.user_id, "active_listing_inventory"])


@pytest.mark.parametrize("regime,gamma", [("randomized", 0), ("targeted", 0), ("targeted", 2)])
def test_logged_features_and_holdout_do_not_expose_private_state(regime, gamma):
    from c2csynth.benchmark import ITEM_DESCRIPTORS
    from c2csynth.dynamic_policy import ACTION_COLUMNS, FeatureEncoder, generate_random_log
    from c2csynth.item_market_benchmark import ItemMarketWorld

    spec = world_spec()
    populations, metadata = population_splits(1, sizes=dict(train=32, validation=16, test=8))
    initial = populations["train"].copy()
    initial["future_outcome"] = 999
    initial["private_engagement"] = 123
    world = ItemMarketWorld(DailyMarketParameters(**spec["parameters"]),
        CouponValidity(**spec["initialization"]["coupon_valid_days"]),
        inventory_zero=spec["inventory_rule"]["zero"], inventory_shape=spec["inventory_rule"]["shape"])
    response = world.response(initial, 7, metadata["calendar_origin"])
    holdout = set(initial.user_id.iloc[:8])
    frame = generate_random_log(response, initial, ITEM_DESCRIPTORS, 7, 7, metadata["calendar_origin"],
        .5, logging=regime, gamma=gamma, targeted_actions=response.targeted_actions, never_assigned=holdout)
    assert not {"future_outcome", "private_engagement", "latent_offset", "item_id"} & set(frame.columns)
    assert not frame.loc[frame.user_id.isin(holdout), ACTION_COLUMNS].to_numpy().any()
    encoder = FeatureEncoder().fit(frame)
    context = encoder.context(frame)
    changed = frame.assign(future_outcome=-999, private_engagement=-123)
    np.testing.assert_array_equal(context, encoder.context(changed))


def test_fidelity_evaluator_on_generated_panel():
    from c2csynth import DailyItemMarket
    from c2csynth.fidelity import MarketPanel, FidelityTolerance, evaluate_market, forward_counts

    populations, metadata = population_splits(1, sizes=dict(train=32, validation=16, test=8))
    initial = populations["train"]
    spec = world_spec()
    parameters = DailyMarketParameters(**spec["parameters"])
    validity = CouponValidity(**spec["initialization"]["coupon_valid_days"])
    requests = np.zeros((21, len(initial), 3), dtype=int)
    env = DailyItemMarket(initial, parameters, validity, seed=17, start_date=metadata["calendar_origin"])
    counts = np.stack([env.step(day).counts for day in requests])
    panel = MarketPanel(initial, pd.date_range(metadata["calendar_origin"], periods=21), requests,
        np.full((21, len(initial)), -1), {d: forward_counts(counts, d) for d in (7, 14)},
        np.zeros((21, len(initial)), dtype=bool))
    report, summary = evaluate_market(panel, parameters, validity, seeds=(17,),
        tolerance=FidelityTolerance(), n_bootstrap=0)
    np.testing.assert_array_equal(report.loc[report.metric.isin(["incidence", "mean_count"]), "error"], 0)
    assert summary["max_user_incidence_ks"] == 0
    assert summary["agreement"]["primary"]["fraction"] == 1
