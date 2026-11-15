"""Generate logs, fit estimators, and evaluate cumulative policy effects.

Use --preset smoke for an end-to-end connection check and --preset paper for
the reported world, population sizes, and training/Monte Carlo settings.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import torch

from c2csynth.dynamic_policy import (
    ACTION_COLUMNS, CLEAR_POLICIES, EVENTS, FeatureEncoder, FixedDescriptors,
    LearnedResponse, MarkovTransition, generate_random_log, observable_initial,
    RATES, ipw_policy_rates, mc_se, responsive_target_mask, rollout_policy,
    score_policy_samples, sequence_arrays,
)
from c2csynth.env import item_coupon_eligible
from c2csynth.gt_compact_residual import CompactResidualGTransformer
from c2csynth.gt_initialization import treatment_context_indices
from c2csynth.item_market import CouponValidity, DailyMarketParameters
from c2csynth.item_market_benchmark import ItemMarketWorld
from c2csynth.vendor.xiong_gt import EmpiricalResidualBank, GTransformerConfig
from c2csynth.benchmark_utils import digest, fitted_loss, load_log, tensor_batch, verify_files
from c2csynth.resources import REFERENCE_WORLD, POPULATION, world_spec, population_splits

ITEM_DESCRIPTORS = FixedDescriptors({
    "buyer_coupon_type_f": "buyer coupon", "buyer_coupon_segment_f": "all",
    "buyer_coupon_event_cnt": 1, "seller_coupon_event_cnt": 1,
    "item_coupon_target_item_cnt": 1, "target_item_avg_price": None, "item_price_available": 0,
})
POLICIES = ("logging_025", "noop")
MARKOV_RUNS = ("markov_same_day", "markov_gbt_same_day")


def is_compact(name):
    return name == "compact_gt_same_day"


def selected_models(args):
    return tuple(args.models)


def clear_protocol(args):
    return True


def evaluation_policies(args):
    return CLEAR_POLICIES


def report_horizons(args):
    return (7, 14, 21)


def uses_holdout_difference(args):
    return args.control_share > 0 and args.logging == "randomized"


def training_runs(args):
    return {**{name: (name, None) for name in MARKOV_RUNS}, **{
        f"{name}_seed{seed}": (name, seed)
        for name in selected_models(args) for seed in args.learning_seeds
    }}


def make_model(name, config, encoder):
    if not is_compact(name):
        raise ValueError(f"Unknown published GT baseline: {name}")
    return CompactResidualGTransformer(config, encoder, exposure_history=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "paper"), default="smoke")
    p.add_argument("--regime", choices=("randomized", "observable", "latent"), default="randomized")
    p.add_argument("--log", type=int, default=1, dest="log_seed")
    p.add_argument("--output", type=Path, required=True, dest="output_root")
    p.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--models", choices=("compact_gt_same_day",), nargs="*",
                   help="Omit for GT in logs 1-3; pass --models with no values for classical baselines only")
    p.add_argument("--oracle-from", type=Path)
    p.add_argument("--stage", choices=("all", "prepare", "train", "evaluate", "report"), default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if not 1 <= a.log_seed <= 10 or a.threads < 1:
        p.error("--log must be 1..10; --threads must be positive")
    a.smoke_test = a.preset == "smoke"
    a.world, a.clear_protocol, a.item_requires_inventory = "item_market", True, True
    a.logging = "randomized" if a.regime == "randomized" else "targeted"
    a.logging_gamma = 2. if a.regime == "latent" else 0.
    a.control_share, a.seed, a.population_seed = .2, 20261111, 2027
    a.learning_seeds, a.treatment_l2 = [2027], .001
    if a.models is None:
        a.models = ["compact_gt_same_day"] if a.log_seed <= 3 else []
    if len(a.models) != len(set(a.models)):
        p.error("Each model may be selected only once")
    a.train_users, a.validation_users, a.test_users = (64, 32, 8) if a.smoke_test else (10000, 5000, 3000)
    a.epochs, a.patience = (2, 2) if a.smoke_test else (30, 10)
    a.oracle_mc, a.model_mc = (2, 2) if a.smoke_test else (128, 32)
    a.effective_batch_size, a.micro_batch_size, a.rollout_batch_size = (32, 8, 8) if a.smoke_test else (512, 32, 3000)
    a.output_root = a.output_root.resolve()
    if a.oracle_from is not None:
        a.oracle_from = a.oracle_from.resolve()
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return a


def experiment_contract(a):
    spec = world_spec()
    metadata = json.loads((POPULATION / "metadata.json").read_text())
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
                if k not in ("resume", "dry_run", "stage")}
    code = Path(__file__).resolve().parent
    inputs = [*(REFERENCE_WORLD.glob("*.json")), *(POPULATION.iterdir())]
    return dict(version=1, settings=settings, models=list(selected_models(a)),
        training_runs={name: list(run) for name, run in training_runs(a).items()},
        policies=list(CLEAR_POLICIES), report_horizons=[7, 14, 21],
        world_id=spec["world_id"], world_sha256=spec["world_sha256"],
        generation=dict(train_days=42, rollout_days=42, report_days=[7, 14, 21],
            logging_probability=.025, log_seed=a.log_seed, control_share=a.control_share,
            initial_population="released initial states and fixed split arrays", calendar_origin=metadata["calendar_origin"]),
        learning=dict(objective="factual CE + standardized log-count MSE; GT treatment penalty .001",
            optimizer="Adam cosine 1e-4 to 1e-5", selection="minimum user-disjoint factual validation loss; no oracle"),
        torch_version=str(torch.__version__),
        code_sha256={str(p.relative_to(code)): digest(p) for p in sorted(code.rglob("*.py"))},
        input_sha256={str(p.relative_to(code)): digest(p) for p in sorted(inputs) if p.is_file()})


def prepare_item_market(a, root):
    spec = world_spec()
    rule = spec["inventory_rule"]
    world = ItemMarketWorld(DailyMarketParameters(**spec["parameters"]),
        CouponValidity(**spec["initialization"]["coupon_valid_days"]),
        inventory_zero=rule["zero"], inventory_shape=rule["shape"])
    populations, metadata = population_splits(a.log_seed,
        sizes=dict(train=a.train_users, validation=a.validation_users, test=a.test_users))
    start_date = metadata["calendar_origin"]
    data_dir = root / "data"
    data_dir.mkdir()
    for offset, name in enumerate(("train", "validation"), start=1):
        say(f"[prepare] {name} logs ({a.regime}, log {a.log_seed})")
        population = populations[name]
        seed = a.log_seed + 10000 * offset
        response = world.response(population, seed, start_date)
        holdout = set(np.random.default_rng([seed, 911]).permutation(population.user_id.to_numpy(dtype=str))[
            :round(a.control_share * len(population))])
        log = generate_random_log(response, population, ITEM_DESCRIPTORS, 42, seed, start_date, .025,
            logging=a.logging, gamma=a.logging_gamma, targeted_actions=response.targeted_actions,
            never_assigned=holdout or None)
        log["holdout"] = log.user_id.astype(str).isin(holdout).astype(int)
        log.to_csv(data_dir / f"{name}.csv.gz", index=False)
    population = populations["test"].sort_values("user_id").reset_index(drop=True)
    observable_initial(population).to_csv(data_dir / "test_initial.csv.gz", index=False)
    joblib.dump({"simulator": world, "population": population, "descriptors": ITEM_DESCRIPTORS},
        data_dir / "oracle_private.pkl.gz", compress=3)
    write_json(root / "data_manifest.json", dict(world="item_market", world_id=spec["world_id"],
        world_sha256=spec["world_sha256"], start_date=start_date, train_days=42, logging_probability=.025,
        logging_regime=a.logging, logging_gamma=a.logging_gamma, item_requires_inventory=True,
        log_seed=a.log_seed, control_share=a.control_share, initial_inventory_rule="concentrated",
        inventory_rule=dict(inventory_zero=rule["zero"], inventory_shape=rule["shape"]),
        coupon_valid_days=asdict(world.validity), descriptor_values=ITEM_DESCRIPTORS.values,
        parameters=asdict(world.parameters),
        files_sha256={p.name: digest(p) for p in sorted(data_dir.iterdir())}))

# === Preserved execution functions from the reported benchmark ===

def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)



def save_torch(path, data):
    temp = path.with_name(path.name + ".tmp")
    torch.save(data, temp)
    temp.replace(path)



def say(message):
    print(message, flush=True)



class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()



def receipt_valid(root, name):
    path = root / f"{name}_completed.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record["experiment_sha256"] != digest(root / "experiment.json"):
        raise ValueError(f"{name}: experiment contract changed")
    for rel, sha in record["files_sha256"].items():
        if not (root / rel).is_file() or digest(root / rel) != sha:
            raise ValueError(f"{name}: completed artifact changed: {rel}")
    return True



def complete(root, name, paths):
    write_json(root / f"{name}_completed.json", {
        "experiment_sha256": digest(root / "experiment.json"),
        "files_sha256": {str(p.relative_to(root)): digest(p) for p in paths}})



def prepare_data(a):
    root = a.output_root
    if receipt_valid(root, "data"):
        say("[reuse] verified shared data")
        return
    if (root / "data_manifest.json").exists():
        manifest = json.loads((root / "data_manifest.json").read_text())
        verify_files(root / "data", manifest["files_sha256"])
    else:
        # A failed prepare may have partial output. Keep it for diagnosis; only
        # regenerate this incomplete stage, never touch completed data/models.
        if (root / "data").exists():
            (root / "data").rename(root / f"incomplete_data_{time.time_ns()}")
        prepare_item_market(a, root)
    manifest = json.loads((root / "data_manifest.json").read_text())
    if (manifest["logging_probability"] != .025 or manifest["train_days"] != 42
            or manifest.get("item_requires_inventory", False) != a.item_requires_inventory):
        raise ValueError("Original logging and 42-day contract mismatch")
    data_world = manifest.get("world", manifest.get("response_config", {}).get("buyer_carryover", "observational"))
    if data_world != a.world:
        raise ValueError("Generated data belong to a different response world")
    if (manifest.get("logging_regime", "randomized"), manifest.get("logging_gamma", 0.)) != (a.logging, a.logging_gamma):
        raise ValueError("Generated data belong to a different logging regime")
    train_frame = load_log(root / "data/train.csv.gz")
    val_frame = load_log(root / "data/validation.csv.gz")
    test_frame = load_log(root / "data/test_initial.csv.gz")
    ids = [set(f.user_id) for f in (train_frame, val_frame, test_frame)]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError("User splits overlap")
    encoder = FeatureEncoder().fit(train_frame)
    sequence_arrays(train_frame, encoder)
    sequence_arrays(val_frame, encoder)
    write_json(root / "feature_encoder.json", encoder.to_dict())
    counts = {"train": len(ids[0]), "validation": len(ids[1]), "test": len(ids[2])}
    if counts != dict(train=a.train_users, validation=a.validation_users, test=a.test_users):
        raise ValueError("User counts differ from protocol")
    audit = dict(users=counts, disjoint=True,
        train_event_rates=train_frame[RATES].mean().to_dict(),
        validation_event_rates=val_frame[RATES].mean().to_dict(),
        train_action_rates=train_frame[ACTION_COLUMNS].mean().to_dict(),
        source_scope="synthetic logs from the released reference initial states",
        smoke_test=a.smoke_test)
    if "holdout" in train_frame:
        audit["holdout_users"] = {name: int(frame.groupby("user_id").holdout.first().sum())
                                  for name, frame in (("train", train_frame), ("validation", val_frame))}
        if any(frame.loc[frame.holdout.eq(1), ACTION_COLUMNS].to_numpy().any() for frame in (train_frame, val_frame)):
            raise ValueError("Factual log assigns a coupon to a holdout user")
    if clear_protocol(a):
        audit["eligibility_and_support"] = {
            name: audit_eligible_log(frame) for name, frame in (("train", train_frame), ("validation", val_frame))
        }
        if not a.smoke_test and not all(
            cell["assigned"] > 0 and cell["unassigned"] > 0
            for cell in audit["eligibility_and_support"]["train"]["responsive_target_support"].values()
        ):
            raise ValueError("Responsive target strata require treated and untreated training rows")
    write_json(root / "data_audit.json", audit)
    complete(root, "data", [root / "data_manifest.json", root / "feature_encoder.json",
        root / "data_audit.json", *sorted((root / "data").iterdir())])



def audit_eligible_log(frame):
    eligible = item_coupon_eligible(frame)
    item = frame["item_coupon_assigned"].to_numpy() == 1
    if (item & ~eligible).any():
        raise ValueError("Factual log contains item assignments without inventory")
    if (frame["sold_cnt_1d"].to_numpy() > np.floor(frame["active_listing_inventory"])).any():
        raise ValueError("Factual log contains sales above pre-action inventory")
    targets = responsive_target_mask(frame, gap=3)
    return dict(user_days=len(frame), item_eligible_user_days=int(eligible.sum()),
                item_assigned=int(item.sum()),
                item_eligible_assignment_rate=float(item.sum() / eligible.sum()) if eligible.any() else None,
                responsive_target_support={col: {
                    "assigned": int((targets[:, i] & frame[col].eq(1)).sum()),
                    "unassigned": int((targets[:, i] & frame[col].eq(0)).sum()),
                } for i, col in enumerate(ACTION_COLUMNS)})



def fit_ipw(a):
    """Weighting baseline on the factual training log; no rollout and no oracle."""
    root = a.output_root
    if receipt_valid(root, "ipw"):
        say("[reuse] verified completed ipw")
        return
    path = root / "ipw"
    path.mkdir(exist_ok=True)
    encoder = FeatureEncoder.from_dict(json.loads((root / "feature_encoder.json").read_text()))
    policies = evaluation_policies(a)
    rates, diagnostics = ipw_policy_rates(load_log(root / "data/train.csv.gz"), encoder, policies)
    temp = path / "estimate.npz.tmp"
    with temp.open("wb") as handle:
        np.savez_compressed(handle, rates=rates, policy_names=np.asarray(policies))
    temp.replace(path / "estimate.npz")
    write_json(path / "summary.json", dict(
        model_variant="ipw", estimator="per-decision self-normalized inverse-probability weighting",
        propensity="one logistic regression per channel on the observed pre-assignment state",
        cohort="factual training users", **diagnostics))
    complete(root, "ipw", [path / "estimate.npz", path / "summary.json"])
    say("[train] IPW estimate computed; effective users at day 42: " + ", ".join(
        f"{k}={v:.0f}" for k, v in diagnostics["effective_users_final_day"].items()))



def fit_holdout_difference(a):
    """Difference in means between the two logged arms; no model, no rollout and no oracle."""
    root = a.output_root
    if receipt_valid(root, "holdout_difference"):
        say("[reuse] verified completed holdout_difference")
        return
    path = root / "holdout_difference"
    path.mkdir(exist_ok=True)
    frame = pd.concat([load_log(root / f"data/{name}.csv.gz") for name in ("train", "validation")])
    frame = frame.sort_values(["user_id", "day"])
    users = frame.user_id.nunique()
    held = frame.holdout.to_numpy(bool).reshape(users, -1)[:, 0]
    cumulative = frame[RATES].to_numpy(float).reshape(users, -1, len(RATES)).cumsum(axis=1)
    arms = cumulative[~held], cumulative[held]
    effect = arms[0].mean(axis=0) - arms[1].mean(axis=0)
    standard_error = np.sqrt(sum(arm.var(axis=0, ddof=1) / len(arm) for arm in arms))
    temp = path / "estimate.npz.tmp"
    with temp.open("wb") as handle:
        np.savez_compressed(handle, cumulative_effect=effect, standard_error=standard_error)
    temp.replace(path / "estimate.npz")
    write_json(path / "summary.json", dict(
        model_variant="holdout_difference", target="logging_025 minus noop",
        estimator="mean cumulative event-days of the assigned arm minus the holdout arm, by day",
        cohort="factual training and validation users",
        users=dict(assigned=int((~held).sum()), holdout=int(held.sum())),
        standard_error="between users, unequal variances"))
    complete(root, "holdout_difference", [path / "estimate.npz", path / "summary.json"])
    say("[train] holdout difference computed; 42-day effect per 100 users: " + ", ".join(
        f"{rate}={100 * effect[-1, k]:+.2f} (SE {100 * standard_error[-1, k]:.2f})" for k, rate in enumerate(RATES)))



def train_markov(a, run_id="markov"):
    root = a.output_root
    if receipt_valid(root, run_id):
        say(f"[reuse] verified completed {run_id}")
        return
    path = root / run_id
    path.mkdir(exist_ok=True)
    encoder = FeatureEncoder.from_dict(json.loads((root / "feature_encoder.json").read_text()))
    train = sequence_arrays(load_log(root / "data/train.csv.gz"), encoder)
    validation = sequence_arrays(load_log(root / "data/validation.csv.gz"), encoder)
    kind = "gbt" if run_id.startswith("markov_gbt") else "logistic"
    history = not run_id.endswith("_same_day")
    model = MarkovTransition(kind, exposure_history=history).fit(train, validation)
    temp = path / "model.pkl.gz.tmp"
    joblib.dump(model, temp, compress=3)
    temp.replace(path / "model.pkl.gz")
    write_json(path / "training_summary.json", dict(model_variant=run_id, seed=None,
        predictors="current observed state/action/descriptor and previous generated events"
                   + ("" if history else ", without the bookkeeping of earlier assignments"),
        objective=("gradient-boosted occurrence models (200 trees, 15 leaves, learning rate 0.05)"
                   if kind == "gbt" else "logistic occurrence models (C=1)")
                  + ", ridge amounts (alpha=10), validation residuals",
        data_manifest_sha256=digest(root / "data_manifest.json")))
    complete(root, run_id, [path / "model.pkl.gz", path / "training_summary.json"])
    say(f"[train] {run_id} transition fitted")



def capture_rng(device):
    state = {"cpu": torch.get_rng_state()}
    if device == "mps":
        state["mps"] = torch.mps.get_rng_state()
    if device == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state



def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])



def verify_initial_null(a, model, encoder, train_frame):
    """Only observed initial states and training residuals; never oracle cache."""
    observed = load_log(a.output_root / "data/test_initial.csv.gz").sort_values("user_id").head(8)
    # Descriptor distribution is immaterial to the null identity. This checker
    # uses training-log descriptors and does not open the oracle/private cache.
    item_world = getattr(a, "world", None) == "item_market"
    descriptors = ITEM_DESCRIPTORS
    residual = encoder.events(train_frame[EVENTS].to_numpy(float)) - model.continuous_head.bias.detach().cpu().numpy()
    bank = EmpiricalResidualBank(torch.as_tensor(residual[:10000], dtype=torch.float32))
    model.eval()
    outputs = []
    for policy in POLICIES:
        response = LearnedResponse(model, encoder, residuals=bank, device=a.device, batch_size=8,
                                   count_only=item_world)
        events, actions = rollout_policy(response, observed, descriptors, policy, 42,
                                         a.seed + 7781, str(observed.date.iloc[0]),
                                         item_requires_inventory=getattr(a, "item_requires_inventory", True))
        outputs.append((events, actions, np.stack(response.event_history, axis=1), response.previous_events))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][2], outputs[1][2])
    np.testing.assert_array_equal(outputs[0][3], outputs[1][3])
    if not outputs[0][1].any() or outputs[1][1].any():
        raise ValueError("Null check must compare different action paths")
    return dict(horizon=42, users=len(observed), same_sampled_event_days=True,
                same_generated_event_histories=True, different_action_paths=True,
                effect_per_user=[0., 0., 0.], initialization_only=True,
                excludes="coupon costs; null assertion concerns purchase/listing/sale events")



def train_model(a, name, *, run_id=None, seed=None):
    root = a.output_root
    run_id = name if run_id is None else run_id
    seed = a.seed if seed is None else seed
    if receipt_valid(root, run_id):
        say(f"[reuse] verified completed {run_id}")
        return
    path = root / run_id
    path.mkdir(exist_ok=True)
    encoder = FeatureEncoder.from_dict(json.loads((root / "feature_encoder.json").read_text()))
    train_frame = load_log(root / "data/train.csv.gz")
    train_data = sequence_arrays(train_frame, encoder)
    val_data = sequence_arrays(load_log(root / "data/validation.csv.gz"), encoder)
    config = GTransformerConfig(dim_continuous=5, categorical_cardinalities=(2, 2, 2),
        dim_action=train_data["context"].shape[-1], hidden_dim=8 if a.smoke_test else 32,
        feedforward_dim=32 if a.smoke_test else 128, num_encoder_layers=1 if a.smoke_test else 3,
        num_attention_heads=2 if a.smoke_test else 4, categorical_head_relu=False,
        batch_size=a.micro_batch_size, max_epochs=a.epochs, early_stopping_patience=a.patience)
    torch.manual_seed(seed)
    model = make_model(name, config, encoder).to(a.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    order_rng = np.random.default_rng(seed)
    history, best, best_epoch, best_state = [], float("inf"), 0, None
    last_path = path / "last.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        order_rng.bit_generator.state = saved["order_rng"]
        restore_rng(saved["torch_rng"])
        history, best, best_epoch = saved["history"], saved["best"], saved["best_epoch"]
        best_state = saved["best_state"]
        say(f"[resume] {run_id} after epoch {len(history)}")
    elif name == "residual_gt" or is_compact(name):
        initialization = model.initialize_from_factual(train_frame, encoder)
        initialization["treatment_context_columns"] = (
            model.treatment_columns if is_compact(name) else treatment_context_indices(encoder)[1])
        write_json(path / "initialization.json", initialization)
        write_json(path / "initial_null_check.json", verify_initial_null(a, model, encoder, train_frame))
        say("[check] residual GT initial 42-day policy difference is exactly zero")
    n = len(train_data["context"])
    n_updates = math.ceil(n / a.effective_batch_size)
    max_updates = a.epochs * n_updates
    for epoch in range(len(history) + 1, a.epochs + 1):
        if history and len(history) - best_epoch >= a.patience:
            break
        started = time.monotonic()
        model.train()
        total = 0.
        regularization_total = 0.
        batches = np.array_split(order_rng.permutation(n), n_updates)
        for update, users in enumerate(batches):
            progress = ((epoch - 1) * n_updates + update) / max(1, max_updates - 1)
            lr = 1e-5 + .5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(users), a.micro_batch_size):
                rows = users[start:start + a.micro_batch_size]
                loss = model.training_loss(*tensor_batch(train_data, rows, a.device))
                regularization = (getattr(a, "treatment_l2", .001) * model.treatment_penalty()
                                  if is_compact(name) else loss.total.new_zeros(()))
                objective = loss.total + regularization
                if not torch.isfinite(objective):
                    raise FloatingPointError(f"{name}: nonfinite training loss")
                (objective * (len(rows) / len(users))).backward()
                total += objective.item() * len(rows)
                regularization_total += regularization.item() * len(rows)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
        model.eval()
        validation = fitted_loss(model, val_data, a.micro_batch_size, a.device)
        if not all(np.isfinite(v) for v in validation.values()):
            raise FloatingPointError(f"{name}: nonfinite validation loss")
        row = dict(epoch=epoch, train_loss=total / n, validation=validation, lr=lr,
                   seconds=time.monotonic() - started)
        if is_compact(name):
            row.update(train_regularization=regularization_total / n,
                       train_factual_loss=(total - regularization_total) / n)
        history.append(row)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if validation["total"] < best:
            best, best_epoch = validation["total"], epoch
            best_state = state
        # Best and latest weights belong to one atomic epoch checkpoint, so an
        # interruption cannot pair a new best.pt with an older optimizer/RNG.
        save_torch(last_path, dict(state_dict=state, optimizer=optimizer.state_dict(),
            order_rng=order_rng.bit_generator.state, torch_rng=capture_rng(a.device),
            history=history, best=best, best_epoch=best_epoch, best_state=best_state))
        write_json(path / "training_history.json", history)
        say(f"[train] {run_id} epoch={epoch}/{a.epochs} val={validation['total']:.6f} "
            f"best={best_epoch} seconds={row['seconds']:.1f}")
    if best_epoch != min(history, key=lambda row: row["validation"]["total"])["epoch"]:
        raise ValueError("Checkpoint/history disagree")
    model.load_state_dict(best_state)
    model.eval()
    residuals = []
    with torch.no_grad():
        for start in range(0, len(val_data["context"]), a.micro_batch_size):
            batch = tensor_batch(val_data, slice(start, start + a.micro_batch_size), a.device)
            predicted = model.predict_prefixes(*batch[:4])
            residuals.append((batch[4] - predicted.continuous_mean).reshape(-1, 5).cpu())
    save_torch(path / "gt.pt", dict(config=asdict(config), state_dict=best_state,
                                   residuals=torch.cat(residuals), model_variant=name))
    summary = dict(model_variant=name, seed=seed, best_epoch=best_epoch, stop_epoch=len(history),
        stop_reason="patience" if len(history) - best_epoch >= a.patience else "max_epochs",
        best_validation_loss=best, parameters=sum(p.numel() for p in model.parameters()),
        config=asdict(config), selection="minimum factual validation CE+MSE; no oracle",
        objective="categorical CE + standardized log1p continuous MSE",
        optimizer="Adam cosine 1e-4 to 1e-5, weight_decay 0", updates_per_epoch=n_updates,
        effective_batch_sizes=sorted({len(b) for b in np.array_split(np.arange(n), n_updates)}),
        micro_batch_size=a.micro_batch_size, total_epoch_seconds=sum(r["seconds"] for r in history),
        data_manifest_sha256=digest(root / "data_manifest.json"))
    if is_compact(name):
        summary.update(treatment_l2=getattr(a, "treatment_l2", .001),
            treatment_inputs="exposure history and current assignment" if model.exposure_history
                             else "current assignment only",
            objective="categorical CE + standardized log1p MSE + treatment_l2 * mean output coefficient squared norm",
            treatment_parameters=model.treatment_categorical.weight.numel()+model.treatment_continuous.weight.numel(),
            treatment_reference=("encoded never-exposed history, " if model.exposure_history else "")
                                + "zero current assignments/details, none coupon categories")
    write_json(path / "training_summary.json", summary)
    files = [path / x for x in ("gt.pt", "training_summary.json", "training_history.json")]
    if name == "residual_gt" or is_compact(name):
        files += [path / "initialization.json", path / "initial_null_check.json"]
    complete(root, run_id, files)



def reuse_oracle(a, manifest, observed, policies, destination):
    """Copy another run's oracle draws once both runs are shown to pose the same question."""
    source = a.oracle_from
    if not receipt_valid(source, "evaluation"):
        raise ValueError("--oracle-from needs a run with a completed evaluation")
    code = [json.loads((root / "experiment.json").read_text())["code_sha256"] for root in (source, a.output_root)]
    if code[0] != code[1]:
        raise ValueError("--oracle-from run was made with other code")
    other = json.loads((source / "data_manifest.json").read_text())
    for key in ("world", "start_date", "parameters", "coupon_valid_days", "initial_inventory_rule",
                "inventory_rule", "item_requires_inventory"):
        if other.get(key) != manifest.get(key):
            raise ValueError(f"--oracle-from run differs in {key}")
    record = json.loads((source / "evaluation/evaluation_manifest.json").read_text())
    if record["policies"] != list(policies) or record["seed"] != a.seed or record["oracle_mc"] < a.oracle_mc:
        raise ValueError("--oracle-from run has other policies, another --seed, or too few oracle draws")
    theirs = load_log(source / "data/test_initial.csv.gz").sort_values("user_id").reset_index(drop=True)
    if not theirs.equals(observed):
        raise ValueError("--oracle-from run evaluates other initial users")
    destination.mkdir(exist_ok=True)
    for rep in range(a.oracle_mc):
        for suffix in ("npz", "json"):  # the receipt last, so a partial copy is simulated again
            name = f"mc_{rep:04d}.{suffix}"
            if not (destination / name).exists():
                shutil.copy2(source / "evaluation/oracle" / name, destination / name)
    say(f"[evaluate] copied {a.oracle_mc} oracle draws from {source}")



def evaluate(a):
    root = a.output_root
    runs = training_runs(a)
    policies = evaluation_policies(a)
    # Check every selected model before loading any oracle state or rewards.
    if not all(receipt_valid(root, name) for name in runs):
        raise ValueError("Complete all selected factual model selections before evaluation")
    if receipt_valid(root, "evaluation"):
        say("[reuse] verified completed evaluation")
        return
    path = root / "evaluation"
    path.mkdir(exist_ok=True)
    manifest = json.loads((root / "data_manifest.json").read_text())
    item_world = manifest.get("world") == "item_market"
    observed = load_log(root / "data/test_initial.csv.gz").sort_values("user_id").reset_index(drop=True)
    cache = joblib.load(root / "data/oracle_private.pkl.gz")
    hidden = cache["population"].copy()
    hidden["user_id"] = hidden.user_id.astype(str)
    hidden = hidden.sort_values("user_id").reset_index(drop=True)
    if hidden.user_id.tolist() != observed.user_id.tolist():
        raise ValueError("Oracle/learner initial populations differ")
    encoder = FeatureEncoder.from_dict(json.loads((root / "feature_encoder.json").read_text()))
    if getattr(a, "oracle_from", None) is not None:
        reuse_oracle(a, manifest, observed, policies, path / "oracle")
    files = []
    for method in ("oracle", *runs):
        destination = path / method
        destination.mkdir(exist_ok=True)
        repeats = a.oracle_mc if method == "oracle" else a.model_mc
        model = bank = None
        if method in MARKOV_RUNS:
            model = joblib.load(root / method / "model.pkl.gz")
        elif method != "oracle":
            saved = torch.load(root / method / "gt.pt", map_location="cpu", weights_only=True)
            model = make_model(runs[method][0], GTransformerConfig(**saved["config"]), encoder).to(a.device)
            model.load_state_dict(saved["state_dict"])
            model.eval()
            bank = EmpiricalResidualBank(saved["residuals"])
        for rep in range(repeats):
            sample = destination / f"mc_{rep:04d}.npz"
            receipt = destination / f"mc_{rep:04d}.json"
            if receipt.exists():
                if not sample.exists() or json.loads(receipt.read_text())["sha256"] != digest(sample):
                    raise ValueError(f"Changed MC checkpoint: {sample}")
            else:
                started = time.monotonic()
                # Pair policy branches; share learner RNG streams across GTs.
                # The oracle uses a separate stream from either learner.
                seed = a.seed + (10000000 if method == "oracle" else 30000000) + 100003 * rep
                events, actions = [], []
                for policy in policies:
                    if method == "oracle":
                        world = cache["simulator"]
                        response = world.response(hidden, seed, manifest["start_date"]) if item_world else world
                    else:
                        response = LearnedResponse(model, encoder, residuals=bank, device=a.device,
                                                   batch_size=a.rollout_batch_size, count_only=item_world)
                    population = hidden if method == "oracle" else observed
                    event, action = rollout_policy(response, population, cache["descriptors"],
                                                   policy, 42, seed, manifest["start_date"],
                                                   item_requires_inventory=manifest.get("item_requires_inventory", False))
                    events.append(event.astype(np.uint8))
                    actions.append(action.astype(np.uint8))
                temp = sample.with_name(sample.name + ".tmp")
                with temp.open("wb") as handle:
                    np.savez_compressed(handle, events=np.stack(events), actions=np.stack(actions),
                                        policy_names=np.asarray(policies), user_ids=observed.user_id.to_numpy(dtype=str))
                temp.replace(sample)
                write_json(receipt, dict(sha256=digest(sample), seed=seed, seconds=time.monotonic() - started))
                if rep == 0 or (rep + 1) % 8 == 0 or rep + 1 == repeats:
                    say(f"[evaluate] {method} MC {rep + 1}/{repeats} last={time.monotonic()-started:.1f}s")
            files += [sample, receipt]
        del model, bank
        if a.device == "mps":
            torch.mps.empty_cache()
    write_json(path / "evaluation_manifest.json", dict(horizon=42, policies=list(policies),
        models=list(runs), item_requires_inventory=manifest.get("item_requires_inventory", False),
        oracle_mc=a.oracle_mc, model_mc=a.model_mc, seed=a.seed,
        oracle_shared_between_models=True, paired_policy_random_streams=True,
        selected_all_models_before_oracle=True,
        oracle_copied_from=str(a.oracle_from) if getattr(a, "oracle_from", None) is not None else None,
        unit="event-days per initial user; reporting also multiplies by 100",
        uncertainty="per-run Monte Carlo uncertainty conditional on world, users and fitted models"))
    complete(root, "evaluation", [path / "evaluation_manifest.json", *files])



def load_samples(path, policies=POLICIES):
    events, actions = [], []
    for filename in sorted(path.glob("mc_*.npz")):
        with np.load(filename) as data:
            if data["policy_names"].tolist() != list(policies):
                raise ValueError("Policy order changed")
            events.append(data["events"])
            actions.append(data["actions"])
    return np.stack(events), np.stack(actions)



def log_estimate_rows(model, policy, reference, horizon, oracle, predicted, sampling_se=None):
    """Report rows of an estimate read off the factual log, without rollouts."""
    rows = []
    for k, outcome in enumerate(RATES):
        actual, se = float(oracle[:, k].mean()), float(mc_se(oracle[:, k]))
        error, zero = abs(float(predicted[k]) - actual), abs(actual)
        rows.append(dict(model=model, run=model, learning_seed=None,
            policy=policy, reference_policy=reference, horizon=horizon, outcome=outcome,
            oracle_effect_per100=actual * 100, predicted_effect_per100=float(predicted[k]) * 100,
            absolute_error_per100=error * 100, zero_absolute_error_per100=zero * 100,
            improvement_vs_zero_percent=100 * (1 - error / zero) if zero else None,
            beats_zero=error < zero, oracle_mc_se_per100=100 * se,
            predicted_mc_se_per100=0., error_mc_se_per100=100 * se))
        if sampling_se is not None:
            rows[-1]["predicted_sampling_se_per100"] = 100 * float(sampling_se[k])
    return rows



def report(a):
    root = a.output_root
    runs = training_runs(a)
    policies = evaluation_policies(a)
    if not receipt_valid(root, "evaluation"):
        raise ValueError("Evaluation incomplete")
    truth, truth_actions = load_samples(root / "evaluation/oracle", policies)
    rows, metrics = [], {}
    for name, (variant, seed) in runs.items():
        events, actions = load_samples(root / "evaluation" / name, policies)
        metrics[name] = {}
        for horizon in report_horizons(a):
            for i, policy in enumerate(policies[:-1]):
                pair = [i, len(policies) - 1]
                m = score_policy_samples(events[:, pair, :, :horizon], truth[:, pair, :, :horizon],
                    actions[:, pair, :, :horizon], truth_actions[:, pair, :, :horizon],
                    policies=(policy, policies[-1]))
                metric_key = f"{policy}_vs_{policies[-1]}:{horizon}" if clear_protocol(a) else str(horizon)
                metrics[name][metric_key] = m
                for outcome, values in m["contrasts"].items():
                    actual = values["cumulative_effect_oracle"]
                    predicted = values["cumulative_effect_predicted"]
                    error, zero = abs(predicted - actual), abs(actual)
                    rows.append(dict(model=variant, run=name, learning_seed=seed,
                        policy=policy, reference_policy=policies[-1], horizon=horizon, outcome=outcome,
                        oracle_effect_per100=actual * 100, predicted_effect_per100=predicted * 100,
                        absolute_error_per100=error * 100, zero_absolute_error_per100=zero * 100,
                        improvement_vs_zero_percent=100 * (1 - error / zero) if zero else None,
                        beats_zero=error < zero, oracle_mc_se_per100=100 * values["oracle_effect_mc_se"],
                        predicted_mc_se_per100=100 * values["predicted_effect_mc_se"],
                        error_mc_se_per100=100 * values["error_mc_se"]))
                    if name == next(iter(runs)):
                        rows.append(dict(model="zero", run="zero", learning_seed=None,
                            policy=policy, reference_policy=policies[-1], horizon=horizon, outcome=outcome,
                            oracle_effect_per100=actual * 100, predicted_effect_per100=0.,
                            absolute_error_per100=zero * 100, zero_absolute_error_per100=zero * 100,
                            improvement_vs_zero_percent=0. if zero else None, beats_zero=False,
                            oracle_mc_se_per100=100 * values["oracle_effect_mc_se"],
                            predicted_mc_se_per100=0., error_mc_se_per100=100 * values["oracle_effect_mc_se"]))
        del events, actions
    if clear_protocol(a):
        if not receipt_valid(root, "ipw"):
            raise ValueError("IPW estimate incomplete")
        with np.load(root / "ipw/estimate.npz") as data:
            if data["policy_names"].tolist() != list(policies):
                raise ValueError("Policy order changed")
            ipw = data["rates"]
        oracle_daily = truth.mean(axis=2)
        for horizon in report_horizons(a):
            for i, policy in enumerate(policies[:-1]):
                oracle = (oracle_daily[:, i, :horizon] - oracle_daily[:, -1, :horizon]).sum(axis=1)
                rows += log_estimate_rows("ipw", policy, policies[-1], horizon, oracle,
                                          (ipw[i, :horizon] - ipw[-1, :horizon]).sum(axis=0))
        if uses_holdout_difference(a):
            if not receipt_valid(root, "holdout_difference"):
                raise ValueError("Holdout difference incomplete")
            with np.load(root / "holdout_difference/estimate.npz") as data:
                effect, sampling_se = data["cumulative_effect"], data["standard_error"]
            i = policies.index("logging_025")
            for horizon in report_horizons(a):
                oracle = (oracle_daily[:, i, :horizon] - oracle_daily[:, -1, :horizon]).sum(axis=1)
                rows += log_estimate_rows("holdout_difference", "logging_025", policies[-1], horizon, oracle,
                                          effect[horizon - 1], sampling_se[horizon - 1])
    path = root / "reports"
    path.mkdir(exist_ok=True)
    write_json(path / "metrics.json", metrics)
    frame = pd.DataFrame(rows).sort_values(["policy", "horizon", "outcome", "run"])
    frame.to_csv(path / "horizon_comparison.csv", index=False)
    final = report_horizons(a)[-1]
    frame.loc[frame.horizon.eq(final)].to_csv(path / f"comparison_{final}days.csv", index=False)
    if clear_protocol(a):
        aggregate = frame.groupby(["policy", "reference_policy", "horizon", "outcome", "model"], as_index=False).agg(
            runs=("run", "size"), oracle_effect_per100=("oracle_effect_per100", "first"),
            oracle_mc_se_per100=("oracle_mc_se_per100", "first"),
            mean_prediction_per100=("predicted_effect_per100", "mean"),
            prediction_seed_sd_per100=("predicted_effect_per100", "std"),
            mean_absolute_error_per100=("absolute_error_per100", "mean"),
            absolute_error_seed_sd_per100=("absolute_error_per100", "std"),
            runs_beating_zero=("beats_zero", "sum"))
        aggregate.to_csv(path / "aggregate.csv", index=False)
    write_json(path / "summary.json", dict(scope=getattr(a, "world", "original response world"), policies=list(policies),
        smoke_test=a.smoke_test, seed=a.seed, training={name: json.loads(
            (root / name / "training_summary.json").read_text()) for name in runs},
        **{f"results_{final}days": [r for r in rows if r["horizon"] == final]},
        uncertainty="MC SE only; winning by point estimate is not proof beyond MC uncertainty"))
    complete(root, "report", sorted(path.iterdir()))
    say(f"[done] {path / f'comparison_{final}days.csv'}")



def main(argv=None):
    a = parse_args(argv)
    torch.set_num_threads(a.threads)
    if a.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable; choose --device cpu or cuda")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    contract = experiment_contract(a)
    if a.dry_run:
        print(json.dumps({k: contract[k] for k in ("settings", "generation", "learning")}, indent=2))
        return
    root = a.output_root
    record = root / "experiment.json"
    if root.exists() and any(root.iterdir()):
        if not a.resume or not record.exists() or json.loads(record.read_text()) != contract:
            raise ValueError("Use a fresh output root or --resume with unchanged code, inputs and settings")
    else:
        root.mkdir(parents=True, exist_ok=True)
        (root / ".gitignore").write_text("*\n!.gitignore\n")
        write_json(record, contract)
    with (root / "workflow.log").open("a", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            try:
                say(f"[run] stage={a.stage} seed={a.seed} device={a.device} smoke={a.smoke_test} "
                    f"models={','.join(selected_models(a))}")
                if a.stage in ("all", "prepare"):
                    prepare_data(a)
                if a.stage != "prepare" and not receipt_valid(root, "data"):
                    raise ValueError("Run the prepare stage first")
                if a.stage in ("all", "train"):
                    if clear_protocol(a):
                        fit_ipw(a)
                        if uses_holdout_difference(a):
                            fit_holdout_difference(a)
                    for run, (name, seed) in training_runs(a).items():
                        if name in MARKOV_RUNS:
                            train_markov(a, run)
                        else:
                            train_model(a, name, run_id=run, seed=seed)
                if a.stage in ("all", "evaluate"):
                    evaluate(a)
                if a.stage in ("all", "evaluate", "report"):
                    report(a)
            except BaseException:
                traceback.print_exc()
                raise



if __name__ == "__main__":
    main()
