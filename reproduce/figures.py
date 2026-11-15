#!/usr/bin/env python3
"""Regenerate the manuscript tables and figures from public aggregates."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
import numpy as np
import pandas as pd
from c2csynth.resources import REFERENCE_WORLD, world_spec

REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "results/paper"
ROOT = REPO / "outputs/paper-figures"
HORIZON = 21
REGIMES = [("randomized", "Randomized"), ("observable", "Observed-state"), ("latent", "Private-state")]
# Keep the saved run identifiers while presenting one configuration per baseline.
TRANSITION_METHODS = [("markov_same_day", "Markov"), ("markov_gbt_same_day", "Tree Markov"),
                      ("compact_gt_same_day", "Compact GT")]
METHODS = [("holdout_difference", "Arm"), ("ipw", "IPW"), *TRANSITION_METHODS]
METHOD_HEADER = " & ".join(label for _, label in METHODS) + r" \\"
POLICY_LABELS = {"logging_025": "Fixed", "responsive_025": "Adaptive"}
RATE_LABELS = {"bcr": "Purchase", "lcr": "Listing", "scr": "Sale"}
CHANNELS = ["purchase", "listing", "sale"]
POLICIES = ["logging_025", "responsive_025", "noop"]
COLORS = ["#245a81", "#a64c32", "#555555"]
OBSERVED = "#2166b5"
SIMULATED = "#14855b"
FIDELITY = ["period_1_fresh", "period_2_fresh_fixed", "period_2_fresh_releveled"]
LABELS = {"logging_025": "Fixed", "responsive_025": "Adaptive"}
RATE_LABELS = {"bcr": "Purchase", "lcr": "Listing", "scr": "Sale"}
CHANNELS = ["purchase", "listing", "sale"]
POLICIES = ["logging_025", "responsive_025", "noop"]
COLORS = ["#245a81", "#a64c32", "#555555"]
OBSERVED = "#2166b5"
SIMULATED = "#14855b"
FIDELITY_FILES = {
    "period_1_fresh": "q2_audit2_tuned_fixed_20seeds_fidelity.csv",
    "period_2_fresh_fixed": "q3_audit2_tuned_fixed_20seeds_fidelity.csv",
    "period_2_fresh_releveled": "q3_audit2_tuned_transfer_20seeds_fidelity.csv",
}
FIDELITY = list(FIDELITY_FILES)
LABELS = ["P1", "P2 fixed", "P2 adjusted"]


def write(name, text):
    (ROOT / "generated" / name).write_text(text + "\n")



def benchmark_results(macro):
    """Tables and in-text numbers of the replicated estimator comparison."""
    summary = pd.read_csv(INPUT / "benchmark_summary.csv")
    lifts = pd.read_csv(INPUT / "benchmark_learned_lifts.csv")
    ipw = pd.read_csv(INPUT / "benchmark_ipw.csv")
    world = json.loads((REFERENCE_WORLD / "parameters.json").read_text())
    summary = summary[summary.model != "zero"]
    # A zero-effect comparison needs an oracle effect that differs from zero.
    summary = summary.assign(detectable=summary.oracle_effect_per100.abs() > 3 * summary.oracle_mc_se_per100)
    excluded = summary.loc[~summary.detectable, ["policy", "outcome"]].drop_duplicates().values.tolist()
    if excluded != [["responsive_025", "lcr"]]:
        raise ValueError(f"Unexpected contrasts without a detectable oracle effect: {excluded}")
    cells = summary.set_index(["regime", "model", "policy", "outcome", "horizon"]).sort_index()

    def wins(regime, model, horizon):
        rows = summary[(summary.regime == regime) & (summary.model == model) & (summary.horizon == horizon)
                       & summary.detectable]
        return (int(rows.logs_beating_zero.sum()), int(rows.logs.sum())) if len(rows) else None

    header = [r"\toprule", "Logging & Day & " + METHOD_HEADER, r"\midrule"]
    lines = [r"\begin{tabular}{@{}ll" + "r" * len(METHODS) + r"@{}}", *header]
    for j, (regime, label) in enumerate(REGIMES):
        for horizon in [7, 14, HORIZON]:
            counts = [wins(regime, model, horizon) for model, _ in METHODS]
            lines.append(f'{label if horizon == 7 else ""} & {horizon} & '
                         + " & ".join("---" if count is None else f"{count[0]}/{count[1]}" for count in counts) + r" \\")
        if j < len(REGIMES) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("revision_wins.tex", "\n".join(lines))

    def contrast_table(name, horizon, value):
        header = [r"\toprule", "Logging & Policy & Outcome & Oracle & " + METHOD_HEADER, r"\midrule"]
        lines = [r"\begin{tabular}{@{}lllr" + "r" * len(METHODS) + r"@{}}", *header]
        for j, (regime, label) in enumerate(REGIMES):
            first = True
            for policy, policy_label in POLICY_LABELS.items():
                for outcome, outcome_label in RATE_LABELS.items():
                    oracle = cells.loc[(regime, "ipw", policy, outcome, horizon)]
                    entries = [value(cells.loc[key]) if (key := (regime, model, policy, outcome, horizon)) in cells.index
                               else "---" for model, _ in METHODS]
                    lines.append(f'{label if first else ""} & {policy_label if outcome == "bcr" else ""} & {outcome_label} & '
                                 f'{oracle.oracle_effect_per100:+.2f} & ' + " & ".join(entries) + r" \\")
                    first = False
            if j < len(REGIMES) - 1:
                lines.append(r"\addlinespace")
        lines += [r"\bottomrule", r"\end{tabular}"]
        write(name, "\n".join(lines))

    for horizon in [14, HORIZON]:
        contrast_table(f"revision_errors_{horizon}.tex", horizon, lambda row: f"{row.mean_absolute_error_per100:.2f}")
    contrast_table("revision_bias.tex", HORIZON,
                   lambda row: f"{row.bias_per100:+.2f} ({row.sd_between_logs_per100:.2f})")

    lift = lifts.groupby(["regime", "model", "coupon"]).learned_logit_shift.agg(["mean", "std", "size"])
    coupons = ["buyer", "lister", "item"]
    specified = [world["buyer_coupon_lift"], world["listing_coupon_lift"], world["item_coupon_lift"]]
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Logging & Estimator & Buyer coupon & Listing incentive & Item coupon & Logs \\",
             r" & & on purchase & on listing & on sale & \\", r"\midrule",
             "Specified lift & & " + " & ".join(f"{value:.2f}" for value in specified) + r" & \\", r"\addlinespace"]
    names = [(model, label) for model, label in TRANSITION_METHODS if model != "markov_gbt_same_day"]
    for j, (regime, label) in enumerate(REGIMES):
        for k, (model, model_label) in enumerate(names):
            entries = [f'{lift.loc[(regime, model, coupon), "mean"]:.2f} ({lift.loc[(regime, model, coupon), "std"]:.2f})'
                       for coupon in coupons]
            lines.append(f'{label if k == 0 else ""} & {model_label} & ' + " & ".join(entries)
                         + f' & {int(lift.loc[(regime, model, "buyer"), "size"])}' + r" \\")
        if j < len(REGIMES) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("revision_lifts.tex", "\n".join(lines))

    effective = ipw.groupby(["regime", "policy"]).effective_users.mean()
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Logging & Fixed & Adaptive & No new assignment \\", r"\midrule"]
    for regime, label in REGIMES:
        lines.append(f"{label} & " + " & ".join(f"{effective.loc[(regime, policy)]:,.0f}" for policy in POLICIES) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("revision_ipw.tex", "\n".join(lines))

    for tag, coupon, value in zip(["Buyer", "Lister", "Item"], coupons, specified):
        fair = lift.loc[("randomized", "markov_same_day", coupon), "mean"]
        private = lift.loc[("latent", "markov_same_day", coupon), "mean"]
        macro[f"RevTrue{tag}"] = f"{value:.2f}"
        macro[f"RevLift{tag}"] = f"{fair:.2f}"
        macro[f"RevLift{tag}Private"] = f"{private:.2f}"
        macro[f"RevLift{tag}Inflation"] = f"{private / fair:.1f}"
    macro["RevGTLift"] = f'{lift.loc[("randomized", "compact_gt_same_day", "buyer"), "mean"]:.2f}'
    macro["RevGTLiftPrivate"] = f'{lift.loc[("latent", "compact_gt_same_day", "buyer"), "mean"]:.2f}'
    for tag, model in [("Arm", "holdout_difference"), ("IPW", "ipw"), ("Markov", "markov_same_day"),
                       ("Tree", "markov_gbt_same_day"), ("GT", "compact_gt_same_day")]:
        count = wins("randomized", model, HORIZON)
        macro[f"RevWin{tag}"] = f"{count[0]} of {count[1]}"
    fixed_purchase = lambda regime, model: cells.loc[(regime, model, "logging_025", "bcr", HORIZON)]
    macro["RevArmSD"] = f'{fixed_purchase("randomized", "holdout_difference").sd_between_logs_per100:.2f}'
    transition_mae = [fixed_purchase("randomized", model).mean_absolute_error_per100
                      for model, _ in TRANSITION_METHODS]
    macro["RevTransitionMAEMin"], macro["RevTransitionMAEMax"] = f"{min(transition_mae):.2f}", f"{max(transition_mae):.2f}"
    macro["RevPrivateMarkov"] = f'{fixed_purchase("latent", "markov_same_day").mean_estimate_per100:+.2f}'



def fidelity_level_table(name, comparisons):
    """Report absolute incidence levels and both distinct uncertainty summaries."""
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Evaluation & Outcome & Observed \% [95\% CI] & Simulated \% [seed range] & Rel. error & User KS \\",
             r"\midrule"]
    for j, (label, frame) in enumerate(comparisons):
        for c, channel in enumerate(CHANNELS):
            data = frame[(frame.window_days == 14) & (frame.channel == channel) & (frame.subset == "all")]
            row = data[data.metric == "incidence"].iloc[0]
            ks = data[data.metric == "user_incidence_ks"].iloc[0].simulated
            observed = f"{100*row.observed:.2f} [{100*row.observed_ci_low:.2f}, {100*row.observed_ci_high:.2f}]"
            simulated = f"{100*row.simulated:.2f} [{100*row.seed_min:.2f}, {100*row.seed_max:.2f}]"
            lines.append(f'{label if c == 0 else ""} & {channel.capitalize()} & {observed} & {simulated} & '
                         f'{100*row.relative_error:+.1f}\\% & {ks:.3f} ' + r"\\")
        if j < len(comparisons) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(name, "\n".join(lines))



def tables_and_numbers():
    frames = {key: pd.read_csv(INPUT / f"{key}_fidelity.csv") for key in FIDELITY}
    summary = json.loads((INPUT / "summary.json").read_text())
    macro = {}
    manifest = json.loads((INPUT / "manifest.json").read_text())
    if not manifest["oracle_matches_current_parameters"]:
        raise ValueError("Oracle and reference parameters must agree")
    macro["RevBenchmarkStatus"] = ""
    workflow = json.loads((INPUT / "workflow.json").read_text())
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Period & Train & Validation & Independent evaluation \\", r"\midrule"]
    cohorts = workflow["primary_cohorts"]
    for period, fields in [("P1", ["train", "validation", "q2_fresh"]),
                           ("P2", ["q3_train", "q3_validation", "q3_fresh"])]:
        cells = [f'{cohorts[name]["users"]:,}' if name in cohorts else "---" for name in fields]
        lines.append(f'{period} & ' + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("revision_cohorts.tex", "\n".join(lines))
    moment_labels = {"multiple_event_rate": "Multiple-event rate", "zero_user_share": "Zero-user share",
                     "user_incidence_sd": "User incidence SD", "weekly_incidence": "Weekly incidence",
                     "lagged_joint_incidence": "Lagged joint incidence"}
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
             r"Moment group & Weight $w_g$ & Scale floor $s_g$ \\", r"\midrule"]
    for key, label in moment_labels.items():
        lines.append(f'{label} & {workflow["loss_weights"][key]:g} & {workflow["loss_floors"][key]:g} ' + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("revision_calibration_loss.tex", "\n".join(lines))
    fidelity_level_table("revision_fidelity_levels.tex", [
        ("P1", frames["period_1_fresh"]),
        ("P2 adjusted", frames["period_2_fresh_releveled"])])
    fidelity_level_table("revision_fidelity_levels_all.tex", list(zip(LABELS, [frames[key] for key in FIDELITY])))
    for stem, key in [("One", "period_1_fresh"), ("Two", "period_2_fresh_releveled")]:
        values = summary[key]
        macro[f"RevFid{stem}Inc"] = f'{100 * values["max_abs_relative_incidence_error"]:.1f}'
        macro[f"RevFid{stem}Count"] = f'{100 * values["max_abs_relative_count_error"]:.1f}'
        macro[f"RevFid{stem}KS"] = f'{values["max_user_incidence_ks"]:.3f}'
    for tag, channel, subset in [("ListSale", "sale", "listing"), ("SaleList", "listing", "sale"),
                                 ("PurchasePurchase", "purchase", "purchase")]:
        frame = frames["period_1_fresh"]
        row = frame[(frame.metric == "lagged_joint_incidence") & (frame.channel == channel)
                    & (frame.subset == subset)].iloc[0]
        macro[f"RevLag{tag}"] = f'{100 * row.relative_error:+.1f}'
    for window in [7, 14]:
        lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
                 r"Evaluation & Outcome & Incidence error (\%) & Count error (\%) & User KS \\", r"\midrule"]
        for j, key in enumerate(FIDELITY):
            for c, channel in enumerate(CHANNELS):
                frame = frames[key]
                sub = frame[(frame.window_days == window) & (frame.channel == channel) & (frame.subset == "all")]
                incidence = sub[sub.metric == "incidence"].iloc[0].relative_error * 100
                count = sub[sub.metric == "mean_count"].iloc[0].relative_error * 100
                ks = sub[sub.metric == "user_incidence_ks"].iloc[0].simulated
                lines.append(f'{LABELS[j] if c == 0 else ""} & {channel.capitalize()} & {incidence:+.1f} & {count:+.1f} & {ks:.3f} ' + r"\\")
            if j < len(FIDELITY) - 1:
                lines.append(r"\addlinespace")
        lines += [r"\bottomrule", r"\end{tabular}"]
        write(f"revision_fidelity_{window}.tex", "\n".join(lines))
    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{7 days} & \multicolumn{3}{c}{14 days} \\",
             r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
             r"Evaluation & Outcome & Inc. err. (\%) & Count err. (\%) & KS & Inc. err. (\%) & Count err. (\%) & KS \\",
             r"\midrule"]
    for j, key in enumerate(FIDELITY):
        for c, channel in enumerate(CHANNELS):
            cells = []
            for window in [7, 14]:
                data = frames[key]
                sub = data[(data.window_days == window) & (data.channel == channel) & (data.subset == "all")]
                for metric in ["incidence", "mean_count", "user_incidence_ks"]:
                    row = sub[sub.metric == metric].iloc[0]
                    cells.append(f"{row.simulated:.3f}" if metric == "user_incidence_ks" else f"{100 * row.relative_error:+.1f}")
            lines.append(f'{LABELS[j] if c == 0 else ""} & {channel.capitalize()} & ' + " & ".join(cells) + r" \\")
        if j < len(FIDELITY)-1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("revision_fidelity_windows.tex", "\n".join(lines))
    with np.load(INPUT / "oracle_aggregate.npz") as data:
        levels = data["events"].cumsum(axis=2) * 100
        difference = levels[:, :2] - levels[:, [2]]
        offers = data["actions"][:, :, :HORIZON].sum(axis=2).mean(axis=0) * 100
    mean = difference.mean(axis=0)
    se = difference.std(axis=0, ddof=1) / np.sqrt(len(difference))
    last = HORIZON - 1
    for p, ptag in enumerate(["F", "A"]):
        for c, ctag in enumerate(["P", "L", "S"]):
            name = f"Rev{ptag}{ctag}"
            macro[name] = f"{mean[p, last, c]:+.2f}"
            macro[name + "SE"] = f"{se[p, last, c]:.2f}"
            macro[name + "Seven"] = f"{mean[p, 6, c]:+.2f}"
            macro[name + "Fourteen"] = f"{mean[p, 13, c]:+.2f}"
            macro[name + "Offers"] = f"{offers[p, c]:.1f}"
            macro[name + "Rel"] = f"{100 * mean[p, last, c] / levels[:, 2, last, c].mean():+.1f}"
    lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             rf"Policy & Outcome & Day 7 & Day 14 & Day {HORIZON} \\", r"\midrule"]
    for p, label in enumerate(["Fixed", "Adaptive"]):
        for c, channel in enumerate(CHANNELS):
            cells = [f"{mean[p,h-1,c]:+.3f} ({se[p,h-1,c]:.3f})" for h in [7, 14, HORIZON]]
            lines.append(f'{label if c == 0 else ""} & {channel.capitalize()} & ' + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("revision_horizons.tex", "\n".join(lines))
    benchmark_results(macro)
    write("revision_numbers.tex", "% Frozen item-market evidence; generated by build_revision_assets.py.\n" +
          "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macro.items()))
    parameters = json.loads((REFERENCE_WORLD / "parameters.json").read_text())
    lines = [r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Parameter & Reference value \\", r"\midrule"]
    for key, value in parameters.items():
        lines.append(key.replace("_", r"\_") + f" & {value:.5g} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("revision_parameters.tex", "\n".join(lines))
    return frames, mean, se, offers



def fidelity_point_ranges(frames):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    fig.subplots_adjust(left=.135, right=.985, bottom=.21, top=.78, wspace=.65)
    extents = []
    for index, (ax, key, title) in enumerate(zip(
            axes, ["period_1_fresh", "period_2_fresh_releveled"],
            ["(a) Period 1", "(b) Period 2, adjusted"])):
        frame = frames[key]
        labels = []
        for c, channel in enumerate(CHANNELS):
            row = frame[(frame.metric == "incidence") & (frame.window_days == 14)
                        & (frame.channel == channel) & (frame.subset == "all")].iloc[0]
            observed_ci = 100 * np.array([row.observed_ci_low-row.observed,
                                         row.observed_ci_high-row.observed])
            seed_range = 100 * np.array([row.seed_min-row.observed, row.seed_max-row.observed])
            delta = 100 * (row.simulated-row.observed)
            y = 2-c
            observed = ax.errorbar(0, y+.10, xerr=np.array([[-observed_ci[0]], [observed_ci[1]]]),
                                  fmt="o", mfc="white", mec=OBSERVED, color=OBSERVED,
                                  ms=4.5, capsize=3, lw=1.3)
            simulated = ax.errorbar(delta, y-.14,
                                    xerr=np.array([[delta-seed_range[0]], [seed_range[1]-delta]]),
                                    fmt="D", color=SIMULATED, ms=4.3, capsize=3, lw=1.3)
            annotation = "0.00 pp" if abs(delta) < .005 else f"{delta:+.2f} pp"
            ax.text(delta, y-.39, annotation, ha="center", va="top",
                    color=SIMULATED, fontsize=8)
            labels.append(f"{channel.capitalize()}\nobserved {100*row.observed:.2f}%")
            extents.extend([*observed_ci, *seed_range])
        ax.set_yticks([2, 1, 0], labels, fontsize=8)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-.64, 2.47)
        ax.axvline(0, color="#8b8b8b", linestyle="--", lw=.8, zorder=0)
        ax.set_xlabel("Simulated minus observed (pp)", fontsize=8.5)
        ax.set_title(title, fontsize=9, loc="left", pad=5)
        ax.tick_params(axis="x", labelsize=8)
    limits = (np.floor((min(extents)-.06)*4)/4, np.ceil((max(extents)+.06)*4)/4)
    for ax in axes:
        ax.set_xlim(*limits)
        ax.set_xticks([-.5, 0, .5])
    fig.legend([observed, simulated],
               ["Observed: user-bootstrap 95% CI", "Simulated: 20-seed mean and range"],
               loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False,
               fontsize=8, handlelength=2.2, columnspacing=2)
    fig.savefig(ROOT / "figures/revision_fidelity.pdf", bbox_inches="tight", pad_inches=.035)
    plt.close(fig)



def mechanism_assets():
    p = json.loads((REFERENCE_WORLD / "parameters.json").read_text())
    data = pd.read_csv(INPUT / "sell_gap.csv")
    contracts = data.selection_contract.dropna().unique()
    if len(contracts) != 1 or "descriptive/noncausal" not in contracts[0]:
        raise ValueError("The sell-to-list diagnostic must retain its population and noncausal status")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5),
                             gridspec_kw={"width_ratios": [.9, 1.05, 1.4]})
    g = np.linspace(0, 84, 300)
    axes[0].plot(g, g / (g+p["sale_gap_saturation_days"]), color=COLORS[1], lw=1.6)
    axes[0].set(xlabel="Preceding sale gap (days)", ylabel="Gap weight", ylim=(0, 1.05),
                xticks=[0, 28, 56, 84])
    d = np.arange(1, 30)
    for gap, color in zip([7, 21, 56], [COLORS[0], COLORS[1], "#7059a4"]):
        response = p["sale_to_listing_gain"]*gap/(gap+p["sale_gap_saturation_days"])
        response = response*p["listing_engagement_retention"]**(d-1)
        axes[1].plot(d, response, color=color, lw=1.5, label=f"Gap {gap} days")
    axes[1].set(xlabel="Days after sale", ylabel="Listing engagement increment",
                ylim=(0, 2), xticks=[1, 7, 14, 28])
    axes[1].legend(frameon=False, fontsize=7.6, loc="upper right", handlelength=1.4)
    order = ["no_sale_7d", "gap_0_6d", "gap_7_13d", "gap_14_27d", "gap_28_55d", "gap_56d_plus"]
    labels = ["No sale\nin 7d", "0-6", "7-13", "14-27", "28-55", "56+"]
    subframes = []
    for period, color, offset, label in [("q2", OBSERVED, -.10, "Period 1"),
                                         ("q3", SIMULATED, .10, "Period 2")]:
        sub = data[data.period.eq(period)].set_index("gap_bin").loc[order]
        axes[2].plot(np.arange(6)+offset, 100*sub.lcr_14d, linestyle="none", marker="o",
                     color=color, ms=3.5, label=label)
        subframes.append(sub)
    axes[2].axvline(.5, color="#cccccc", linestyle=":", lw=.7)
    axes[2].set_xticks(range(6), labels, fontsize=7.5)
    axes[2].set(xlabel="Preceding sale gap (days)", ylabel="14-day listing incidence (%)",
                ylim=(0, 20), yticks=[0, 5, 10, 15, 20])
    axes[2].legend(frameon=False, fontsize=7.6, loc="upper right", handlelength=1)
    for ax, letter in zip(axes, "abc"):
        ax.set_title(f"({letter})", loc="left", fontsize=9, pad=3)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.tick_params(axis="y", labelsize=8)
        if ax is not axes[2]:
            ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(ROOT / "figures/revision_sell_to_list.pdf", bbox_inches="tight", pad_inches=.035)
    plt.close(fig)
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Sale history & P1 user-days & P1 incidence (\%) & P2 user-days & P2 incidence (\%) \\",
             r"\midrule"]
    for j, key in enumerate(order):
        label = "No sale in previous 7 days" if j == 0 else labels[j] + " days"
        cells = [f"{int(sub.loc[key, 'n_rows']):,} & {100*sub.loc[key, 'lcr_14d']:.3f}" for sub in subframes]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("revision_sell_gap_counts.tex", "\n".join(lines))
    half_life = np.log(.5)/np.log(p["listing_engagement_retention"])
    write("revision_mechanism_numbers.tex", f"\\newcommand{{\\RevListingHalfLife}}{{{half_life:.1f}}}")

    # Isolated, deterministic mechanism diagnostics: no fitting or empirical claims.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    d = np.arange(1, 29)
    initial_q = (1-p["satiation_retention"])*p["purchase_satiation"]
    satiation = initial_q*p["satiation_retention"]**(d-1)
    engagement = p["purchase_engagement_share"]*initial_q*p["purchase_engagement_retention"]**(d-1)
    for curve, label, color, style in [(engagement, "Buying engagement", COLORS[0], "-"),
                                       (-satiation, "Negative satiation", COLORS[1], "--"),
                                       (engagement-satiation, "Net purchase logit", COLORS[2], "-")]:
        axes[0].plot(d, curve, label=label, color=color, linestyle=style, lw=1.4)
    axes[0].set(xlabel="Days after purchase", ylabel="Purchase logit contribution",
                xticks=[1, 7, 14, 28])
    axes[0].axhline(0, color="#aaaaaa", lw=.6)
    axes[0].legend(frameon=False, fontsize=7.3, loc="lower right")
    fast, count, last = 0., 0, 0
    responses = []
    days = np.arange(1, 29)
    for day in days:
        fatigue = fast+p["cumulative_fatigue"]*np.log1p(count)*p["fatigue_recovery_days"]/(day-last+p["fatigue_recovery_days"])
        responses.append(1/(1+fatigue))
        delivery = int(day <= 7)
        fast = p["fatigue_retention"]*fast+p["fatigue_increment"]*delivery
        count += delivery
        if delivery:
            last = day
    axes[1].axvspan(.5, 7.5, color="#eeeeee", zorder=0)
    axes[1].plot(days, responses, color=SIMULATED, lw=1.5)
    axes[1].plot(days[:7], responses[:7], linestyle="none", marker="o", color=SIMULATED, ms=3)
    axes[1].set(xlabel="Day", ylabel="Remaining coupon logit contribution",
                ylim=(.45, 1.05), xlim=(.5, 28), xticks=[1, 7, 14, 21, 28])
    axes[1].text(18, .52, "No further deliveries", fontsize=8, ha="center", color=COLORS[2])
    for ax, letter in zip(axes, "ab"):
        ax.set_title(f"({letter})", loc="left", fontsize=9)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.tick_params(labelsize=8)
    fig.tight_layout(w_pad=2)
    fig.savefig(ROOT / "figures/revision_response_mechanisms.pdf", bbox_inches="tight", pad_inches=.035)
    plt.close(fig)

    rct = json.loads((INPUT / "item_coupon_rct.json").read_text())
    for key, heading, digits in [("opportunity", "Allocation opportunity", 3),
                                 ("interval", "Interval (days)", 2)]:
        entries = rct[key]
        levels = entries["index"] if key == "opportunity" else entries["days"]
        lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
                 heading + " & " + " & ".join(map(str, levels)) + r" \\", r"\midrule",
                 "Normalized sale-rate contrast & " + " & ".join(f"{v:.{digits}f}" for v in entries["normalized_effect"]) + r" \\",
                 r"\bottomrule", r"\end{tabular}"]
        write(f"revision_rct_{key}.tex", "\n".join(lines))



def domain_and_workflow_figures():
    fig = plt.figure(figsize=(7.1, 3.95))
    roles = fig.add_axes([.015, .53, .97, .45])
    roles.set(xlim=(0, 10), ylim=(0, 3.5))
    roles.axis("off")
    roles.text(.03, 3.35, "(a)", fontsize=10)
    roles.add_patch(FancyBboxPatch((.15, .06), 9.65, 2.3, boxstyle="round,pad=.03,rounding_size=.08",
                                  facecolor="#fafbfc", edgecolor="#aab1b8", lw=.8))
    roles.text(5.95, 2.17, "One marketplace user", ha="center", fontsize=9, color=COLORS[2])
    nodes = [(.35, "Buyer\nPurchase"), (2.65, "Lister\nNew listing"),
             (5.2, "Available\nitems"), (8.0, "Seller\nItem sale")]
    for x, label in nodes:
        roles.add_patch(Rectangle((x, 1.12), 1.65, .72, facecolor="white", edgecolor="#46515b", lw=.8))
        roles.text(x+.825, 1.48, label, ha="center", va="center", fontsize=9.5)
    for x, label in [(1.175, "Buyer coupon"), (3.475, "Listing incentive"),
                     (8.825, "Item coupon\n(one listed item)")]:
        roles.text(x, 2.93, label, ha="center", va="center", fontsize=9, color=COLORS[0])
        roles.add_patch(FancyArrowPatch((x, 2.59), (x, 1.88), arrowstyle="-|>",
                                       mutation_scale=10, lw=.9, color=COLORS[0]))
    roles.add_patch(FancyArrowPatch((4.34, 1.48), (5.16, 1.48), arrowstyle="-|>", mutation_scale=11, lw=.9))
    roles.text(4.75, 1.75, "next day", ha="center", fontsize=7.7)
    roles.add_patch(FancyArrowPatch((6.9, 1.48), (7.96, 1.48), arrowstyle="-|>", mutation_scale=11, lw=.9))
    roles.text(7.4, .93, "Sold item leaves stock", ha="center", fontsize=7.8)
    roles.plot([8.825, 8.825, 3.475, 3.475], [1.08, .32, .32, .80], color=COLORS[1], lw=.9)
    roles.add_patch(FancyArrowPatch((3.475, .80), (3.475, 1.08), arrowstyle="-|>",
                                   mutation_scale=10, lw=.9, color=COLORS[1]))
    roles.text(6.15, .49, "Selling experience raises future listing propensity", ha="center",
               fontsize=8, color=COLORS[1])

    ax = fig.add_axes([.015, .015, .97, .47])
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    ax.text(.03, 3.75, "(b)", fontsize=10)
    boxes = [(0.1, 2.05, 2.0, "Observed history\nAvailable items"),
             (2.65, 2.05, 1.7, "Feasible\ncoupon policy"),
             (4.95, 2.05, 2.0, "Purchase / listing\nItem-level sales"),
             (7.5, 2.05, 2.35, "Next-day history\nand inventory")]
    for x,y,w,label in boxes:
        ax.add_patch(Rectangle((x,y),w,1.15,facecolor="#f3f4f5",edgecolor="#46515b",lw=.8))
        ax.text(x+w/2,y+.575,label,ha="center",va="center",fontsize=9.5)
    for a,b in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((a[0]+a[2]+.04,2.63),(b[0]-.05,2.63),arrowstyle="-|>",mutation_scale=12,lw=1))
    ax.plot([8.68,8.68,1.1,1.1],[2.02,.4,.4,1.45],color=COLORS[0],lw=1)
    ax.add_patch(FancyArrowPatch((1.1,1.45),(1.1,2.02),arrowstyle="-|>",mutation_scale=12,color=COLORS[0]))
    ax.text(5,.58,"Each policy continues from its own generated state",ha="center",fontsize=10,color=COLORS[0])
    ax.text(5.95,3.75,"Private engagement, satiation, and fatigue",ha="center",fontsize=10)
    ax.add_patch(FancyArrowPatch((5.95,3.48),(5.95,3.23),arrowstyle="-|>",mutation_scale=10,lw=.8))
    ax.text(5.95,1.6,"Realized events update the private state",ha="center",fontsize=9)
    fig.savefig(ROOT/"figures/revision_closed_loop.pdf",bbox_inches="tight",pad_inches=.03)
    plt.close(fig)

    fig = plt.figure(figsize=(7.1, 3.9))
    ax = fig.add_axes([.02, .02, .96, .96])
    ax.set(xlim=(0, 10), ylim=(0, 7))
    ax.axis("off")
    def box(x, y, w, h, label, color="#f3f4f5"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#46515b", lw=.8))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=9.2)
    box(2.35, 5.95, 5.3, .9, "Period 1 training and validation\nJoint fit of behavior and initial inventory")
    left = ["Empirical fidelity\nP1 fixed; P2 fixed or adjusted",
            "Replay observed delivery requests\nGenerate outcomes and update states",
            "Independent evaluation users\nCompare 7- and 14-day summaries"]
    right = ["Same Period 1 parameters\nGenerate factual policy logs",
             "Fit estimators on factual logs\nKeep oracle information separate",
             "Each target policy evolves its own state\nCompare estimated effects with oracle"]
    ax.plot([5, 5], [5.91, 5.44], color="#46515b", lw=.9)
    ax.plot([2.4, 7.6], [5.44, 5.44], color="#46515b", lw=.9)
    for x, labels, color in [(.15, left, "#f2f7fb"), (5.35, right, "#f6f6f0")]:
        center = x+2.25
        ax.add_patch(FancyArrowPatch((center, 5.44), (center, 4.98), arrowstyle="-|>", mutation_scale=11, lw=.9))
        for y, label in zip([3.9, 2.2, .5], labels):
            box(x, y, 4.5, 1.05, label, color)
            if y > .5:
                ax.add_patch(FancyArrowPatch((center, y-.04), (center, y-.61), arrowstyle="-|>",
                                             mutation_scale=11, lw=.9))
    fig.savefig(ROOT/"figures/revision_workflow.pdf", bbox_inches="tight", pad_inches=.04)
    plt.close(fig)



def figures(frames, mean, se):
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fidelity_point_ranges(frames)
    mechanism_assets()
    domain_and_workflow_figures()
    shown = ["period_1_fresh", "period_2_fresh_fixed", "period_2_fresh_releveled"]
    labels = ["P1 fresh", "P2 fresh\nfixed", "P2 fresh\nadjusted"]
    fig, axes = plt.subplots(1,3,figsize=(7.1,2.5))
    for j,(metric,title) in enumerate([("incidence","Incidence error (%)"),("mean_count","Count error (%)"),("user_incidence_ks","User-level KS")]):
        ax=axes[j]
        for c,channel in enumerate(CHANNELS):
            values=[]
            for key in shown:
                frame=frames[key]
                row=frame[(frame.metric==metric)&(frame.window_days==14)&(frame.channel==channel)&(frame.subset=="all")].iloc[0]
                values.append(row.simulated if metric=="user_incidence_ks" else row.relative_error*100)
            ax.plot(np.arange(3)+(c-1)*.12,values,marker=["o","s","^"][c],lw=0,color=COLORS[c],ms=5,label=channel.capitalize())
        ax.set_xticks(range(3),labels,fontsize=8)
        ax.set_ylabel(title,fontsize=9)
        ax.axhline(0,color="#aaaaaa",lw=.7)
        ax.grid(axis="y",alpha=.16)
        if j==2: ax.set_ylim(bottom=0)
    handles, names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,ncol=3,loc="upper center",frameon=False,bbox_to_anchor=(.5,1.04))
    fig.tight_layout(rect=(0,0,1,.92),w_pad=1)
    fig.savefig(ROOT/"figures/revision_fidelity_diagnostics.pdf",bbox_inches="tight",pad_inches=.03)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.15))
    for c,ax in enumerate(axes):
        for p,label in enumerate(["Fixed","Adaptive"]):
            days=range(1,HORIZON+1)
            ax.plot(days,mean[p,:HORIZON,c],label=label,color=COLORS[p],lw=1.4)
            ax.fill_between(days,mean[p,:HORIZON,c]-1.96*se[p,:HORIZON,c],mean[p,:HORIZON,c]+1.96*se[p,:HORIZON,c],color=COLORS[p],alpha=.15,lw=0)
        ax.axhline(0,color="#777777",lw=.7)
        ax.set_xticks([1,7,14,HORIZON])
        ax.set_xlabel("Day",fontsize=9)
        ax.set_title(CHANNELS[c].capitalize(),fontsize=10)
        ax.grid(axis="y",alpha=.16)
    axes[0].set_ylabel("Cumulative effect\n(event-days / 100 users)",fontsize=9)
    axes[2].legend(frameon=False,fontsize=8,loc="best")
    fig.tight_layout(w_pad=1)
    fig.savefig(ROOT/"figures/revision_oracle_effects.pdf",bbox_inches="tight",pad_inches=.03)
    plt.close(fig)



def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.output.resolve()
    (ROOT / "generated").mkdir(parents=True, exist_ok=True)
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)
    world_spec()
    frames, mean, se, offers = tables_and_numbers()
    figures(frames, mean, se)
    print(json.dumps({"output": str(ROOT), "tables": len(list((ROOT / "generated").glob("*.tex"))),
        "figures": len(list((ROOT / "figures").glob("*.pdf"))),
        "oracle_day21": mean[:, 20].tolist(), "mc_se": se[:, 20].tolist()}, indent=2))


if __name__ == "__main__":
    main()
