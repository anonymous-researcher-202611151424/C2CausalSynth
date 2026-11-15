# C2CausalSynth

A daily closed-loop simulator for sequential coupon policy evaluation in
consumer-to-consumer marketplaces. The environment generates purchases,
listings, and item sales; maintains inventory and behavioral history; and
supports buyer, listing, and item coupons.

For a short review path, see [the reviewer guide](docs/reviewer_guide.md).

**The P1 fidelity evaluation and the sequential benchmark use the same frozen
behavioral parameters, coupon-response parameters, and inventory rule.**
The reference world is stored once under
`src/c2csynth/resources/worlds/reference/`. Targeting changes the assignment
policy, not the outcome model.

## Install and use the simulator

Use Python 3.12 or newer. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/quickstart.py
```

The example uses 256 released initial users and runs for 21 days. For a custom
policy, obtain the observed state, produce a three-column request array, and
call `step`:

```python
import numpy as np
from c2csynth import DailyItemMarket, DailyMarketParameters, CouponValidity
from c2csynth.resources import population_splits, world_spec

populations, metadata = population_splits(1)
world = world_spec()
env = DailyItemMarket(
    populations["test"], DailyMarketParameters(**world["parameters"]),
    CouponValidity(**world["initialization"]["coupon_valid_days"]),
    seed=2027, start_date=metadata["calendar_origin"],
)
state = env.observed_users()
requests = np.zeros((len(state), 3), dtype=int)  # buyer, listing, item
events = env.step(requests)
print(events.counts.sum(axis=0))  # purchase, listing, sale counts
```

The simulator also accepts a user-supplied initial table and parameter values.
The supplied benchmark population has empirically derived baseline summaries
and synthetic initial inventory. All subsequent benchmark events are generated.

## Run the 21-day policy benchmark

Install the full, pinned experiment environment:

```bash
python -m pip install -r requirements-paper.txt
python -m pip install -e .
```

First run a small complete check, including log generation, classical
baselines, Compact GT training, oracle simulation, and scoring:

```bash
python -m c2csynth.benchmark --preset smoke --output outputs/smoke --device cpu
```

For one full paper log:

```bash
python -m c2csynth.benchmark --preset paper --regime randomized --log 1 \
  --output outputs/paper/randomized/log01 --device auto
```

For the complete replicated benchmark:

```bash
python -m c2csynth.suite --preset paper --output outputs/paper --device auto
```

The suite first computes a shared oracle in randomized log 4, then executes
ten logs in each of three regimes. Compact GT is fitted on logs 1–3; the other
baselines run on all ten. The suite resumes verified completed stages. For a
single run, append `--resume` to continue it with unchanged settings.

The factual training logs cover 42 days. Policy effects are reported at
**7, 14, and 21 days**, using the same underlying 42-day rollouts. Paper settings
use 10,000 training, 5,000 validation, and 3,000 evaluation users, 128 oracle
replicates, and 32 learner-rollout replicates. The smoke preset uses smaller
cohorts, model size, and compute budgets; it is a connection check, not a paper
result. Both presets load exactly the same reference world.

Outputs include `reports/horizon_comparison.csv`, fitted-model selection
records, Monte Carlo samples, and hashes of code and inputs. Suite aggregates
are written to `summary/`. See [the benchmark protocol](docs/benchmark.md) for
policy definitions, estimators, metrics, information boundaries, and expected
reproduction limits.

## Reproduce the paper's figures and tables

```bash
python -m pip install -e '.[figures]'
python reproduce/figures.py --output outputs/paper-figures
python reproduce/verify_results.py
```

These commands use the frozen population-level inputs in `results/paper/`.
They recompute the reported aggregate summaries and regenerate numerical
tables and figures without retraining. They are separate from the full
benchmark command above.

## Included data and reproduction scope

- `src/c2csynth/resources/population/`: the union of 27,000 initial users needed
  by the replicated benchmark, 11 required columns, and fixed split arrays.
  Original identifiers, actual dates, category names, future observed outcomes,
  and identifier mappings are excluded.
- `src/c2csynth/resources/worlds/reference/`: shared P1 parameters, inventory
  distribution, initialization contract, and integrity hashes.
- `src/c2csynth/resources/worlds/period2_adjusted/`: the separately labeled
  period-2 recalibration used only for its fidelity comparison.
- `results/paper/`: aggregate fidelity, oracle, empirical-motivation, and
  benchmark outputs used by the paper.

The full synthetic policy benchmark can be rerun from this distribution.
Empirical calibration and user-level empirical fidelity comparisons require
restricted operational records, which are not distributed. The public
fidelity aggregates support numerical inspection and figure regeneration,
not a new fit against those records. See [data scope](docs/data.md) and
[calibration and fidelity](docs/fidelity.md).

## Tests and integrity

```bash
python -m pip install -e '.[benchmark,test]'
python -m pytest
python reproduce/verify_release.py
```

`MANIFEST.json` and `SHA256SUMS` identify the frozen distribution files.
Tests exercise state transitions, the fixed world and split contracts,
and the separation between observed learner features and oracle information.
The core API and paper benchmark can be used without access to the private
research checkout.

## License and attribution

MIT; see [LICENSE](LICENSE). The G-Transformer architecture is attributed in
[NOTICE.md](NOTICE.md). The Compact GT baseline is an adaptation with a
current-assignment component and shared observed-state bookkeeping.
