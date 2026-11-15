from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from c2csynth.item_market import CouponValidity, DailyItemMarket, DailyMarketParameters


def market(stock, **kwargs):
    users = pd.DataFrame({"user_id": np.arange(len(stock)), "active_listing_inventory": stock})
    return DailyItemMarket(users, **kwargs)


def logit(probability):
    return np.log(probability) - np.log1p(-probability)


def test_item_targeting_and_stock_follow_individual_events():
    env = market([0, 1, 5], parameters=DailyMarketParameters(listing_intercept=-1.5, sale_intercept=-1.5), seed=42)
    request = np.array([[0, 1, 2], [0, 1, 4], [0, 1, 2]])
    control = env.expected_day()
    treated = env.expected_day(request)
    increases = treated.item_sale_probability > control.item_sale_probability
    assert increases.sum() == 3
    assert treated.event_probability[0, 2] == 0
    old_stock = env.inventory()
    events = env.step(request)
    assert set(env.coupons()["coupon_name"]) == {"lister coupon", "item coupon"}
    np.testing.assert_array_equal(events.delivered[:, 2], [0, 1, 2])
    np.testing.assert_array_equal(np.sort(events.targeted_item_ids), treated.eligible_item_ids[increases])
    assert not (set(events.created_item_ids) & set(events.sold_item_ids))
    assert np.all(events.counts[:, 2] <= old_stock)
    np.testing.assert_array_equal(env.inventory(), old_stock + events.counts[:, 1] - events.counts[:, 2])

    all_sold = set(events.sold_item_ids)
    for _ in range(8):
        old_stock = env.inventory()
        events = env.step(request)
        assert not (all_sold & set(events.sold_item_ids))
        all_sold.update(events.sold_item_ids)
        np.testing.assert_array_equal(env.inventory(), old_stock + events.counts[:, 1] - events.counts[:, 2])
    assert env.items().loc[env.items().sold_day >= 0, "item_id"].is_unique


def test_conditional_expectations_agree_with_one_day_sampling():
    size = 40_000
    env = market(np.full(size, 3), parameters=DailyMarketParameters(
        purchase_intercept=-2.5, listing_intercept=-2.8, sale_intercept=-2.7,
        purchase_extra_count=0.5, listing_extra_count=0.7), seed=93)
    requests = np.tile([1, 1, 2], (size, 1))
    expected = env.expected_day(requests)
    assert env.coupons().empty  # Inspecting an action must not issue it.
    observed = env.step(requests).counts
    np.testing.assert_allclose(observed.mean(axis=0), expected.mean_count.mean(axis=0), atol=0.008)
    np.testing.assert_allclose((observed > 0).mean(axis=0), expected.event_probability.mean(axis=0), atol=0.008)


def test_coupon_validity_redemption_and_event_mediated_purchase_feedback():
    params = DailyMarketParameters(purchase_intercept=-2.7, buyer_coupon_lift=1.5)
    env = market(np.zeros(2_000, dtype=int), parameters=params, validity=CouponValidity(buyer=2), seed=42)
    control = deepcopy(env)
    action = np.tile([1, 0, 0], (env.n_users, 1))
    assert np.all(env.expected_day(action).event_probability[:, 0] > control.expected_day().event_probability[:, 0])
    with_offer = env.step(action)
    without_offer = control.step()
    purchased_only_with_offer = (with_offer.counts[:, 0] > 0) & (without_offer.counts[:, 0] == 0)
    did_not_buy = (with_offer.counts[:, 0] == 0) & (without_offer.counts[:, 0] == 0)
    assert purchased_only_with_offer.any() and did_not_buy.any()
    after = env.expected_day().event_probability[:, 0]
    baseline = control.expected_day().event_probability[:, 0]
    assert np.all(after[purchased_only_with_offer] < baseline[purchased_only_with_offer])
    assert np.all(after[did_not_buy] > baseline[did_not_buy])  # Day two of the offer.
    coupons = env.coupons()
    assert set(coupons["coupon_name"]) == {"buyer coupon"}
    redeemed = coupons["redeemed_day"].to_numpy()
    np.testing.assert_array_equal(redeemed >= 0, with_offer.counts[:, 0] > 0)
    assert np.all(env.engagement[purchased_only_with_offer, 0] > control.engagement[purchased_only_with_offer, 0])
    assert not any("engagement" in name for name in env.observed_users().columns)

    bought_day_two = env.step().counts[:, 0] > 0
    control_bought_day_two = control.step().counts[:, 0] > 0
    no_purchases = did_not_buy & ~bought_day_two & ~control_bought_day_two
    assert no_purchases.any()
    np.testing.assert_array_equal(env.expected_day().event_probability[no_purchases, 0],
                                  control.expected_day().event_probability[no_purchases, 0])
    # Same organic probability, but a repeated offer is attenuated.
    assert np.all(env.expected_day(action).event_probability[no_purchases, 0]
                  < control.expected_day(action).event_probability[no_purchases, 0])


def test_a_sale_updates_listing_engagement_without_a_coupon_lag():
    params = DailyMarketParameters(sale_intercept=-1.5)
    stocked = market(np.full(1_000, 2), parameters=params, seed=10)
    empty = market(np.zeros(1_000, dtype=int), parameters=params, seed=10)
    sales = stocked.step().counts[:, 2] > 0
    empty.step()
    assert sales.any() and (~sales).any()
    assert np.all(stocked.expected_day().event_probability[sales, 1]
                  > empty.expected_day().event_probability[sales, 1])
    np.testing.assert_array_equal(stocked.engagement[~sales, 1], empty.engagement[~sales, 1])


def test_delivered_coupons_lower_engagement_on_the_other_side():
    params = DailyMarketParameters(
        purchase_intercept=-2.5, listing_intercept=-2.5, sale_intercept=-2.5, buyer_coupon_lift=0.8,
        listing_coupon_lift=0.6, item_coupon_lift=1.0, sale_to_listing_gain=0.35,
        buyer_coupon_listing_drag=0.4, listing_coupon_purchase_drag=0.2)
    size = 300
    retention = params.listing_engagement_retention

    def action(*request):
        return np.tile(request, (size, 1))

    def pair(validity=None):
        return [market(np.full(size, 3), parameters=params, validity=validity, seed=11) for _ in range(2)]

    def same_feedback(left, right):
        return ((left.counts[:, [0, 2]] > 0) == (right.counts[:, [0, 2]] > 0)).all(axis=1)

    env, reference = pair()
    control = env.expected_day()
    buyer = env.expected_day(action(1, 0, 0))
    np.testing.assert_allclose(logit(buyer.event_probability[:, 0]) - logit(control.event_probability[:, 0]), 0.8)
    np.testing.assert_allclose(logit(buyer.event_probability[:, 1]) - logit(control.event_probability[:, 1]), -0.4)
    np.testing.assert_allclose(logit(buyer.item_sale_probability) - logit(control.item_sale_probability), -0.35 * 0.4)
    lister = env.expected_day(action(0, 1, 0))
    np.testing.assert_allclose(logit(lister.event_probability[:, 0]) - logit(control.event_probability[:, 0]), -0.2)
    np.testing.assert_allclose(logit(lister.event_probability[:, 1]) - logit(control.event_probability[:, 1]), 0.6)
    np.testing.assert_array_equal(lister.item_sale_probability, control.item_sale_probability)
    item = env.expected_day(action(0, 0, 2))
    np.testing.assert_array_equal(item.event_probability[:, :2], control.event_probability[:, :2])

    # The loss is part of the engagement state: it outlasts the one-day
    # coupon and recovers at the retention rate.
    same = same_feedback(env.step(action(1, 0, 0)), reference.step())
    assert same.any()
    np.testing.assert_array_equal(env.engagement[same, 0], reference.engagement[same, 0])
    np.testing.assert_allclose(env.engagement[same, 1] - reference.engagement[same, 1], -0.4 * retention)
    np.testing.assert_allclose(logit(env.expected_day().event_probability[same, 1])
                               - logit(reference.expected_day().event_probability[same, 1]), -0.4 * retention)
    same &= same_feedback(env.step(), reference.step())
    assert same.any()
    np.testing.assert_allclose(env.engagement[same, 1] - reference.engagement[same, 1], -0.4 * retention**2)

    # A repeated delivery is attenuated by fatigue, as its lift is.
    def listing_shift(world):
        return (logit(world.expected_day(action(1, 0, 0)).event_probability[:, 1])
                - logit(world.expected_day().event_probability[:, 1]))

    fatigue = params.fatigue_increment * params.fatigue_retention
    np.testing.assert_allclose(listing_shift(reference), -0.4)
    np.testing.assert_allclose(listing_shift(env), -0.4 / (1 + fatigue))

    # A request during an active coupon is not a delivery and moves nothing.
    held, _ = pair(CouponValidity(buyer=3))
    unredeemed = held.step(action(1, 0, 0)).counts[:, 0] == 0
    assert unredeemed.any()
    np.testing.assert_array_equal(listing_shift(held)[unredeemed], 0)

    # Item coupons do not move the owner's purchase engagement.
    owner, plain = pair()
    owner.step(action(0, 0, 2))
    plain.step()
    np.testing.assert_array_equal(owner.engagement[:, 0], plain.engagement[:, 0])

    # Without drags a coupon leaves the other side unchanged.
    separate = market(np.full(size, 3), parameters=DailyMarketParameters(), seed=11)
    np.testing.assert_array_equal(separate.expected_day(action(1, 0, 0)).event_probability[:, 1],
                                  separate.expected_day().event_probability[:, 1])
    np.testing.assert_array_equal(separate.expected_day(action(0, 1, 2)).event_probability[:, 0],
                                  separate.expected_day().event_probability[:, 0])


def test_engagement_returns_to_baseline_at_side_specific_rates():
    users = pd.DataFrame({"user_id": np.arange(200), "buy_cnt_7d": 3, "list_cnt_7d": 3})
    slow = DailyMarketParameters(purchase_engagement_retention=0.9, listing_engagement_retention=0.95)
    fast = replace(slow, listing_engagement_retention=0.75)
    a, b = DailyItemMarket(users, slow, seed=3), DailyItemMarket(users, fast, seed=3)
    start = a.engagement.copy()
    assert np.all(start > 0)
    first, second = a.step(), b.step()
    np.testing.assert_array_equal(first.counts, second.counts)
    no_sale = first.counts[:, 2] == 0
    assert no_sale.any()
    np.testing.assert_array_equal(a.engagement[:, 0], b.engagement[:, 0])
    np.testing.assert_allclose(a.engagement[no_sale, 1] - b.engagement[no_sale, 1], 0.2 * start[no_sale, 1])

    # Parameter files written with one retention set both sides.
    legacy = DailyMarketParameters(engagement_retention=0.9)
    assert legacy.purchase_engagement_retention == legacy.listing_engagement_retention == 0.9
    assert "engagement_retention" not in asdict(legacy)
    common = replace(slow, engagement_retention=0.88)
    assert common.purchase_engagement_retention == common.listing_engagement_retention == 0.88


def test_a_sale_after_a_long_gap_raises_listing_engagement_more():
    size = 400
    recent = np.arange(size) % 2 == 1
    users = pd.DataFrame({"user_id": np.arange(size), "active_listing_inventory": 30,
                          "days_since_last_sale": np.where(recent, 1.0, 999.0)})
    flat = DailyMarketParameters(sale_intercept=-1.5, sale_user_sd=0.0, sale_to_listing_gain=0.6)
    weighted = replace(flat, sale_gap_saturation_days=7.0)
    a, b = DailyItemMarket(users, flat, seed=4), DailyItemMarket(users, weighted, seed=4)
    sold = a.step().counts[:, 2] > 0
    np.testing.assert_array_equal(b.step().counts[:, 2] > 0, sold)
    assert (sold & recent).any() and (sold & ~recent).any()
    lost = a.engagement[:, 1] - b.engagement[:, 1]
    np.testing.assert_allclose(lost[sold & recent], 0.6 * (1 - 1 / 8))
    np.testing.assert_allclose(lost[sold & ~recent], 0.6 * (1 - 999 / 1006))
    np.testing.assert_array_equal(lost[~sold], 0)


def test_recently_listed_items_sell_more_easily():
    params = DailyMarketParameters(listing_intercept=-1.5, sale_intercept=-4.0, fresh_item_sale_lift=0.9)
    env = market(np.full(300, 2), parameters=params, seed=8)
    plain = market(np.full(300, 2), parameters=replace(params, fresh_item_sale_lift=0.0), seed=8)
    np.testing.assert_array_equal(env.step().counts, plain.step().counts)  # Initial stock has no known age.
    plan, base = env.expected_day(), plain.expected_day()
    np.testing.assert_array_equal(plan.eligible_item_ids, base.eligible_item_ids)
    listed_today = env.items().loc[plan.eligible_item_ids, "listed_day"].to_numpy() == 0
    assert listed_today.any() and (~listed_today).any()
    shift = logit(plan.item_sale_probability) - logit(base.item_sale_probability)
    np.testing.assert_allclose(shift[listed_today], 0.9)
    np.testing.assert_array_equal(shift[~listed_today], 0)

    # After a week on sale an item is as hard to sell as the initial stock.
    for _ in range(7):
        env.step()
    plan = env.expected_day()
    items = env.items().loc[plan.eligible_item_ids]
    frame = pd.DataFrame({"owner": items.user_id.to_numpy(), "aged": items.listed_day.to_numpy() <= 0,
                          "initial": items.listed_day.to_numpy() < 0, "logit": logit(plan.item_sale_probability)})
    aged = frame[frame.aged].groupby("owner")
    assert (aged.initial.nunique() == 2).any()
    np.testing.assert_allclose(aged.logit.min(), aged.logit.max())
    by_age = frame.groupby(["owner", "aged"]).logit.mean().unstack().dropna()
    assert len(by_age)
    np.testing.assert_allclose(by_age[False] - by_age[True], 0.9)


def test_cumulative_fatigue_recovers_with_time_since_the_last_delivery():
    params = DailyMarketParameters(buyer_coupon_lift=1.0, fatigue_increment=0.0, cumulative_fatigue=0.5,
                                   fatigue_recovery_days=10.0)
    env = market(np.zeros(50, dtype=int), parameters=params, seed=2)
    offer = np.tile([1, 0, 0], (50, 1))

    def lift():
        return logit(env.expected_day(offer).event_probability[:, 0]) - logit(env.expected_day().event_probability[:, 0])

    np.testing.assert_allclose(lift(), 1.0)
    env.step(offer)
    observed = []
    for days in (1, 10, 30):
        while env.day < days:
            env.step()
        observed.append(lift())
        np.testing.assert_allclose(observed[-1], 1.0 / (1 + 0.5 * np.log(2) * 10 / (days + 10)))
    assert np.all(observed[0] < observed[1]) and np.all(observed[1] < observed[2]) and np.all(observed[2] < 1.0)
    env.step(offer)  # A second delivery adds to the count and restarts the recovery.
    np.testing.assert_allclose(lift(), 1.0 / (1 + 0.5 * np.log(3) * 10 / 11))


def test_lift_scales_with_private_propensity_on_the_coupon_side():
    size = 500
    params = DailyMarketParameters(
        purchase_intercept=-3.0, listing_intercept=-3.0, sale_intercept=-1.5, buyer_coupon_lift=0.5,
        listing_coupon_lift=0.5, item_coupon_lift=0.5, latent_response_slope=0.4, listing_coupon_purchase_drag=0.3)
    env = market(np.full(size, 2), parameters=params, seed=6)

    def action(*request):
        return np.tile(request, (size, 1))

    def shifts(request):
        control, treated = env.expected_day(), env.expected_day(request)
        return (logit(treated.event_probability[:, :2]) - logit(control.event_probability[:, :2]),
                logit(treated.item_sale_probability) - logit(control.item_sale_probability))

    private = env.latent_logging_scores()  # Engagement and satiation start at zero here.
    buyer = shifts(action(1, 0, 0))[0][:, 0]
    np.testing.assert_allclose(buyer, 0.5 * np.exp(0.4 * private[:, 0]))
    assert buyer.max() > 2 * buyer.min()
    item = shifts(action(0, 0, 1))[1]
    owner = env.items().loc[env.expected_day().eligible_item_ids, "user_id"].astype(int).to_numpy()
    assert (item != 0).sum() == size
    np.testing.assert_allclose(item[item != 0], 0.5 * np.exp(0.4 * private[owner[item != 0], 2]))
    # A lister coupon on the same day lowers purchase engagement, so the buyer coupon lifts less.
    together = shifts(action(1, 1, 0))[0][:, 0]
    np.testing.assert_allclose(together, 0.5 * np.exp(0.4 * (private[:, 0] - 0.3)) - 0.3)

    # After a sale the listing side is engaged and a lister coupon lifts more.
    sold = env.step().counts[:, 2] > 0
    assert sold.any() and (~sold).any()
    lister = shifts(action(0, 1, 0))[0][:, 1]
    private = env.latent_logging_scores()
    np.testing.assert_allclose(lister, 0.5 * np.exp(0.4 * private[:, 1]))
    assert lister[sold].mean() > 1.3 * lister[~sold].mean()


def test_latent_propensities_can_be_correlated_across_channels():
    stock = np.zeros(20_000, dtype=int)

    def correlations(value):
        world = market(stock, parameters=DailyMarketParameters(latent_correlation=value), seed=1)
        return np.corrcoef(world.latent_logging_scores(), rowvar=False)[np.triu_indices(3, 1)]

    np.testing.assert_allclose(correlations(0.0), 0, atol=0.03)
    np.testing.assert_allclose(correlations(0.6), 0.6, atol=0.03)


def test_parameter_and_request_constraints():
    defaults = DailyMarketParameters()
    assert defaults.sale_to_listing_gain > max(
        defaults.buyer_coupon_lift, defaults.listing_coupon_lift, defaults.item_coupon_lift)
    with pytest.raises(ValueError, match="buyer_coupon_listing_drag"):
        DailyMarketParameters(buyer_coupon_listing_drag=-0.1)
    with pytest.raises(ValueError, match="buyer_coupon_lift"):
        DailyMarketParameters(buyer_coupon_lift=-0.1)
    with pytest.raises(ValueError, match="engagement_retention"):
        DailyMarketParameters(engagement_retention=1.0)
    with pytest.raises(ValueError, match="validity"):
        CouponValidity(item=0)
    env = market([1])
    with pytest.raises(ValueError, match="integer item counts"):
        env.step(np.array([[1, 0, 0.5]]))
    assert env.day == 0 and env.coupons().empty
