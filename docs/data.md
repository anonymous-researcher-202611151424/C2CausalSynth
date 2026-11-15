# Released data

The initial-state table contains one baseline row per released user. It is a
restricted projection of the empirical initialization inputs, not a release
of original operational event histories. Benchmark trajectories after day
zero are simulator-generated. No differential-privacy guarantee is claimed.

## Initial columns

- `user_id`: new, ordered identifiers `u00000`, `u00001`, …; no original-ID
  mapping is included.
- `device_os`, `region`: opaque category codes. Code ordering preserves the
  one-hot column ordering of the reported estimators; semantic labels are not
  included. The codes do not describe a user's location in readable form.
- `days_since_signup`: the baseline tenure feature used by the estimators.
- `buy_cnt_7d`, `buy_cnt_30d`: baseline purchase counts.
- `list_cnt_7d`, `list_cnt_30d`: baseline listing counts.
- `sold_cnt_7d_pre`, `sold_cnt_30d_pre`: baseline sale counts.
- `active_listing_inventory`: synthetic initial stock drawn once with the
  shared, published concentrated-inventory rule and seed 42.

The date origin is artificial and preserves weekday. No exact original date
is included. The loader supplies the common cohort labels, derives sale
recency as 3/18/999 days from the 7/30-day sale summaries, and uses a prior
sale gap of 30 days. These are initialization conventions, not recovered
individual sale dates.

## Splits and ordering

`splits.npz` contains training and validation row indices for each of ten logs,
plus the single common evaluation split. Each log uses 10,000 training and
5,000 validation users. The 3,000 evaluation users are disjoint from every
training/validation split. Training membership can differ across logs.

Split arrays preserve selection order, and public IDs preserve the sort order
used by the simulation. This matters for random streams and holdout-arm
selection. Initial stock was drawn before the original cohort was subsetted;
loading a split must not redraw it. Replacing IDs by a hash or independently
resampling the split is not the reported experiment.

## Oracle and observed information

Policies and learners use observed behavioral summaries, inventory, exposure
history, calendar features, and supplied attributes. Private engagement,
latent offsets, internal item identities, and counterfactual outcomes are
excluded from the learner feature list. The oracle samples private states
internally from the same world and random-seed contract. No individual private
state from operational records is needed or distributed.

Generated `outputs/` may contain user-level synthetic logs, model checkpoints,
and evaluator caches; these are runtime products and are gitignored. Frozen
paper results contain population-level or log-level aggregates only.
