# Sequential policy benchmark

## Shared world

All experiments use the reference world's behavior, coupon response, inventory
rule, and one-day coupon validity. Fidelity replays observed requests; benchmark
policies generate requests from their own evolving histories. Their assignment
mechanisms differ, while the reference outcome model is fixed.

Each day follows pre-action observed state → delivered assignments → generated
events → updated inventory and history. Items listed today become available
tomorrow. Item coupons and sales require inventory. Purchases are not matched
to another simulated seller; the external side of the transaction is implicit.

## Target policies

- **Fixed** (`logging_025`): independently assign each channel with daily
  probability 0.025; item coupons require inventory.
- **Adaptive** (`responsive_025`): the same probability with at least three
  days since the previous same-channel assignment. Buyer coupons additionally
  require no purchase in the previous seven days; listing incentives require
  a sale in the previous three days; item coupons require inventory.
- **No new assignment** (`noop`): no coupons, with natural events and state
  updates continuing normally.

Both active policies are contrasted with no new assignment, conditional on
the same initial evaluation cohort. Every policy branch evolves its own state.

## Factual logs

For each regime, ten independent logs vary fitting users and simulated
trajectories. The evaluation cohort and oracle draws remain fixed. Twenty
percent of logged users are randomly reserved for no assignment throughout.

- `randomized`: active users follow Fixed.
- `observable`: +1 logit for the Adaptive target stratum, then recenter to mean
  assignment probability 0.025 among eligible users.
- `latent`: additionally add twice the world's private logging score, then
  recenter to the same mean probability.

Private targeting is a stress test with unobserved confounding. Its effects
are not guaranteed identifiable from observed histories.

## Estimators

- **Markov** (`markov_same_day`): logistic occurrence models and ridge count
  components in transformed coordinates, with validation residual sampling.
- **Tree Markov** (`markov_gbt_same_day`): boosted-tree occurrence models with
  the same observed-state bookkeeping.
- **Compact GT** (`compact_gt_same_day`): a behavioral-history Transformer
  and a centered current-assignment component. Earlier assignment bookkeeping
  is hidden from the base and response component; previous generated behavior
  remains in the state. Hidden size 32, three layers, four heads, treatment L2
  0.001, Adam cosine learning rate 1e-4→1e-5, effective batch 512, micro-batch
  32, at most 30 epochs, patience 10. The seed is 2027 for each of the first
  three logs in each regime. Selection minimizes factual validation loss.
- **IPW**: estimated channel assignment probabilities clipped to [0.001, 0.5],
  cumulative self-normalized weights, truncated at the day's 99th percentile.
- **Arm** (`holdout_difference`): randomized active-arm versus holdout mean
  difference, for Fixed only. This estimates the logging-cohort contrast and
  is compared with the evaluation-cohort oracle as a sampling reference.
- **Zero**: always predict zero effect.

The `same_day` implementation suffix identifies the published configuration;
it does not remove behavioral history or make the task a one-step evaluation.
Count-only item-market outputs use zero GMV placeholders in the shared learner
interface; GMV is not an outcome of this benchmark.

## Metrics and uncertainty

The reported outcome is cumulative purchase/listing/sale **event-days per 100
initial users**, not cumulative transaction amounts. Evaluate at days 7, 14,
and 21. Report the estimated contrast, oracle contrast, MAE, RMSE, bias,
between-log standard deviation, zero-effect comparison, and IPW effective
sample size.

Oracle estimates use 128 Monte Carlo replicates; learned rollouts use 32.
Policies share random streams within a method, while oracle and learner
streams differ. MC uncertainty is conditional on the world, cohort, and fitted
model. Between-log spread adds variation from fitting/log generation, not
world-calibration uncertainty. Figure bands are pointwise mean ±1.96 MC SE.

## Reproduction

Use Python 3.12 and `requirements-paper.txt` for the closest numerical match.
Generation and oracle paths preserve the released world, initial-state values,
ordering, and seed convention. Neural optimization may vary across CPU, MPS,
CUDA, BLAS implementations, and library versions; bitwise retraining equality
across hardware is not asserted. Frozen numerical aggregates are independently
recomputed by `reproduce/verify_results.py`.

The paper suite is computationally substantial: 30 logs, nine Compact GT fits,
one shared 128-replicate oracle, and repeated learner rollouts. Use `--preset
smoke` to check the full pipeline first. `--stage prepare/train/evaluate/report`
allows staged execution. Every stage verifies its recorded hashes before reuse.
