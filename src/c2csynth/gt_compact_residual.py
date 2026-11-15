"""Action-blind GT plus a centered, additive treatment/history adjustment.

The adjustment uses the current observable exposure summaries, not another
Transformer. No intercept or state interactions; signed coefficients are learned
for all three channels and outcomes. A never-exposed history gives exactly zero
adjustment even after fitting. A no-action day after exposure need not be zero.

With ``exposure_history=False`` the adjustment reads today's assignment only. A
day without an assignment then has exactly zero adjustment, whatever was
assigned before; an earlier coupon acts on later days only through the events it
generated. The base never sees exposure bookkeeping in either variant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from c2csynth.dynamic_policy import (
    ACTION_COLUMNS, CATEGORY_COLUMNS, DETAIL_COLUMNS, RATES, SALE_INDEX, STATE_COLUMNS,
)
from c2csynth.env import item_coupon_eligible
from c2csynth.gt_initialization import treatment_context_indices, initialize_factual_heads
from c2csynth.vendor.xiong_gt import XiongGTransformer

# Log-odds removed from the sale head on days without pre-action inventory.
# The inventory rule is public bookkeeping: such a day has no sale by
# construction, so its cross-entropy term and its sampling probability vanish.
SALE_GATE_LOGIT = 30.


def encoded_no_treatment(encoder):
    """Encode the raw reference; standardized zero is NOT the no-treatment value."""
    raw = {name: 0. for name in STATE_COLUMNS + ACTION_COLUMNS + DETAIL_COLUMNS}
    for channel in ('buyer', 'seller', 'item'):
        raw[f'{channel}_days_since_exposure'] = 999.
    raw.update({name: 'none' if name.startswith('buyer_coupon_') else 'unknown'
                for name in CATEGORY_COLUMNS})
    return encoder.context(pd.DataFrame([raw]))[0]


class CompactResidualGTransformer(XiongGTransformer):
    def __init__(self, config, encoder, exposure_history=True):
        if config.categorical_head_relu or tuple(config.categorical_cardinalities) != (2, 2, 2):
            raise ValueError('Compact GT requires three binary unrestricted logit heads')
        super().__init__(config)
        self.exposure_history = exposure_history
        hidden, _ = treatment_context_indices(encoder)
        indices, self.treatment_columns = treatment_context_indices(encoder, exposure_history)
        reference = encoded_no_treatment(encoder)
        if len(reference) != config.dim_action:
            raise ValueError('Encoder/config context width mismatch')
        offset = config.dim_categorical + config.dim_continuous
        self.register_buffer('treatment_history_indices', torch.tensor(indices) + offset)
        self.register_buffer('treatment_reference', torch.from_numpy(reference[indices].copy()))
        # The encoder appends the sale-feasibility bit as the last context column.
        self.feasibility_index = offset + config.dim_action - 1
        width = offset + config.dim_action + config.dim_static
        mask = torch.ones(width)
        # Exposure bookkeeping stays hidden from the base even when the
        # adjustment does not read it.
        mask[torch.tensor(hidden) + offset] = 0
        self.register_buffer('baseline_history_mask', mask)
        # One log-odds coefficient vector per binary outcome; no redundant
        # trainable shift of both class logits. Continuous units are standardized.
        self.treatment_categorical = nn.Linear(len(indices), 3, bias=False)
        self.treatment_continuous = nn.Linear(len(indices), config.dim_continuous, bias=False)
        self.zero_treatment_heads()

    @torch.no_grad()
    def zero_treatment_heads(self):
        self.treatment_categorical.weight.zero_()
        self.treatment_continuous.weight.zero_()

    @staticmethod
    def current_features(history_features, padding_mask):
        if padding_mask is None:
            return history_features[:, -1]
        indices = (~padding_mask).sum(1) - 1
        return history_features[torch.arange(len(indices), device=indices.device), indices]

    def treatment_features(self, history_features, padding_mask):
        last = self.current_features(history_features, padding_mask)
        return last[:, self.treatment_history_indices] - self.treatment_reference

    def _categorical_prediction(self, history_features, padding_mask):
        base = super()._categorical_prediction(history_features * self.baseline_history_mask, padding_mask)
        delta = self.treatment_categorical(self.treatment_features(history_features, padding_mask))
        feasible = self.current_features(history_features, padding_mask)[:, self.feasibility_index]
        gate = (feasible - 1.) * SALE_GATE_LOGIT
        shifts = [delta[:, k] + gate if k == SALE_INDEX else delta[:, k] for k in range(len(base))]
        return tuple(logits + torch.stack((torch.zeros_like(shift), shift), -1)
                     for logits, shift in zip(base, shifts, strict=True))

    def _continuous_prediction(self, history_features, next_categorical_features, padding_mask):
        base = super()._continuous_prediction(history_features * self.baseline_history_mask,
                                              next_categorical_features, padding_mask)
        return base + self.treatment_continuous(self.treatment_features(history_features, padding_mask))

    def treatment_penalty(self):
        # Average squared L2 norm over output coefficient vectors, not features.
        return (self.treatment_categorical.weight.square().sum()
                + self.treatment_continuous.weight.square().sum()) / (3 + self.config.dim_continuous)

    @torch.no_grad()
    def initialize_from_factual(self, train_frame, encoder):
        note = initialize_factual_heads(self, train_frame, encoder)
        self.zero_treatment_heads()
        # The sale head models feasible days only, so its constant starts at the
        # factual no-action sale rate among days with pre-action inventory.
        control = train_frame.loc[train_frame[ACTION_COLUMNS].eq(0).all(axis=1)]
        feasible = control.loc[item_coupon_eligible(control)]
        sale = RATES[SALE_INDEX]
        probability = (float(feasible[sale].sum()) + .5) / (len(feasible) + 1)
        head = self.categorical_heads[SALE_INDEX]
        head.bias.copy_(torch.tensor([np.log1p(-probability), np.log(probability)],
                                     dtype=head.bias.dtype, device=head.bias.device))
        note.update(sale_feasibility='sale head conditioned on pre-action inventory >= 1; zero probability otherwise',
                    feasible_no_action_rows=len(feasible),
                    initial_feasible_sale_probability=probability)
        note.update(treatment_structure='centered additive linear heads, no intercept or sign constraints',
                    treatment_inputs='exposure history and current assignment' if self.exposure_history
                    else 'current assignment only',
                    treatment_features=len(self.treatment_reference),
                    treatment_parameters=sum(p.numel() for module in
                        (self.treatment_categorical, self.treatment_continuous) for p in module.parameters()))
        return note
