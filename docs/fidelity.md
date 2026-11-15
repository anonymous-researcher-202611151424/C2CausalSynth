# Calibration and empirical fidelity

## One reference world

`src/c2csynth/resources/worlds/reference/` contains the single P1 world used
for empirical fidelity and the sequential benchmark. It includes all 30
behavioral/coupon parameters, the concentrated-inventory rule, and the
initialization conventions. Both stages use one-day coupon validity. Fidelity
replays observed requests; the benchmark generates policy-specific requests.
The assignment process changes between these tasks; the reference outcome
model does not.

Period 2 has a fixed-world comparison and a separately labeled adjusted
comparison. `worlds/period2_adjusted/parameters.json` changes only the three
channel intercepts, purchase/listing positive-count means, and sale-propensity
heterogeneity. It is not selected by, or used in, the policy benchmark.

## Fitting and selection

Empirical training and validation users are disjoint. The independent
evaluation sample was drawn after model and transfer selection were frozen.
The sizes are recorded in `results/paper/workflow.json`: P1 has 9,437 training,
5,685 validation, and 5,715 independent evaluation users; P2 has 17,045,
5,682, and 5,690, respectively. These empirical cohorts serve a different
purpose from the synthetic estimator-training splits released here.

Bounded moment matching supplies an initial fit, followed by coordinate
refinement on rescaled parameters with common random numbers. The searched
parameter names, moment weights, and normalization floors are included in
`workflow.json`. Search covers event levels and heterogeneity, positive-count
means, behavioral dynamics, satiation, sale-to-listing feedback, latent
correlation, freshness, and the initial inventory distribution. P1 fitting
uses two simulation seeds and validation uses five. Final P2 refinement uses
four fitting seeds, damped incidence/count moment-ratio updates, and 20
validation seeds. Selection minimizes the primary validation loss.

There are six targets per primary group: three outcomes × 7/14-day windows.
The primary loss is the sum of three group means of squared normalized error:

- Incidence error divided by `max(0.05 * abs(observed), 0.001)`.
- Mean-count error divided by `max(0.10 * abs(observed), 0.002)`.
- Seed-mean user-incidence KS divided by `0.05`.

Refinement adds grouped secondary moment losses. Each squared moment error
is divided by `max(abs(observed), group_floor)^2`; the group mean is multiplied
by its recorded weight. The secondary groups are multiple-event rate,
zero-user share, user-incidence standard deviation, weekly incidence, and
lagged joint incidence. Delivery coverage adds
`0.25 * sum((1 - mean_delivered / requested)^2)` over channels with requests.
This term captures inability to replay requested deliveries under inventory
and coupon constraints. Delivery-window outcome moments and campaign
treatment–control outcome contrasts are excluded from fitting.

## Evaluation and released evidence

Fidelity uses 20 simulation seeds. Missing and censored observed forward
windows are excluded, with the same user/window support used in every draw.
User-level KS compares distributions of each user's incidence averaged over
available window starts; the KS statistics are then averaged across seeds.
Observed intervals use 200 user-cluster bootstrap resamples, retaining each
sampled user's complete series. Simulated ranges span the 20 seeds.

The three `*_fidelity.csv` files contain observed and simulated aggregate
moments, relative errors, observed intervals, simulation
ranges, and tolerance checks. `summary.json` records the maxima used in the
paper. `reproduce/verify_results.py` recomputes these summary values, and
`reproduce/figures.py` recreates the tables and figures.
Weekly subsets use relative block labels; their original calendar dates are
not included. Numerical values are preserved.

`c2csynth.fidelity` exposes the panel schema, window aggregation, moment
measurement, and fidelity evaluation for a caller-supplied panel. The caller
must supply the initial population with fixed inventory, daily requests,
observed forward-window arrays with unavailable entries marked as `NaN`,
and any required cohort/support filtering. This API includes no operational
database reader.

Operational delivery histories and observed future outcomes are not released.
Consequently, the public package supports inspection and regeneration of
the published empirical aggregates, but not an independent refit or
recomputation of those aggregates from individual operational records. The
synthetic policy benchmark is fully runnable from the released initial states.
