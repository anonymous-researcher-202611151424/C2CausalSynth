"""Paper-faithful core of the Xiong et al. G-Transformer.

This module is an independent reproduction of the architecture and algorithms
described in Xiong et al. (MLHC 2024); it is not official author code.  It
implements the parts that distinguish G-Transformer from a conventional
sequence-to-sequence Transformer:

* a categorical encoder followed by a separate continuous encoder;
* teacher forcing of the observed next categorical value during training;
* one independent Transformer example for every observed prefix, retaining
  only the final representation from each example; and
* Monte Carlo simulation that samples categorical values first and adds a draw
  from holdout residuals to each continuous conditional mean.

Inputs follow the g-computation alignment ``(L_t, A_t) -> L_{t+1}``.  The
categorical part of ``L_t`` is supplied as concatenated one-hot vectors.  This
keeps the core generic while allowing the CVSim adapter to use its single
binary pulmonary-edema indicator without a dataset-specific dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


ResidualSampling = Literal["marginal", "joint"]
PositionEncoding = Literal["sinusoidal", "none"]
PrefixExecution = Literal["batched_independent", "serial"]


@dataclass(frozen=True)
class GTransformerConfig:
    """Architecture and published CVSim training defaults.

    Xiong et al. report three encoder layers, hidden dimension 128, batch size
    16, Adam with learning rate ``1e-4``, at most 50 epochs, early-stopping
    patience 10, and ``CosineAnnealingWarmRestarts(T_0=10,
    eta_min=1e-5)`` for CVSim.  The paper does not report the number of heads,
    dropout, or positional encoding.  This reproduction therefore exposes
    those choices and conservatively defaults to four heads, dropout 0.1, and
    the sinusoidal encoding from the original Transformer.

    ``categorical_cardinalities`` gives one class count per categorical
    covariate.  Their concatenated one-hot dimension is available as
    :attr:`dim_categorical`.
    """

    dim_continuous: int
    categorical_cardinalities: tuple[int, ...]
    dim_action: int
    dim_static: int = 0

    hidden_dim: int = 128
    num_encoder_layers: int = 3
    num_attention_heads: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.1
    positional_encoding: PositionEncoding = "sinusoidal"
    max_sequence_length: int = 128
    categorical_head_relu: bool = True
    residual_sampling: ResidualSampling = "marginal"
    prefix_execution: PrefixExecution = "batched_independent"

    categorical_loss_weight: float = 1.0
    continuous_loss_weight: float = 1.0

    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_epochs: int = 50
    early_stopping_patience: int = 10
    scheduler_t0: int = 10
    scheduler_t_mult: int = 1
    scheduler_eta_min: float = 1e-5

    def __post_init__(self) -> None:
        if self.dim_continuous <= 0:
            raise ValueError("dim_continuous must be positive.")
        if not self.categorical_cardinalities:
            raise ValueError("At least one categorical covariate is required.")
        if any(cardinality < 2 for cardinality in self.categorical_cardinalities):
            raise ValueError("Every categorical cardinality must be at least two.")
        if self.dim_action <= 0:
            raise ValueError("dim_action must be positive.")
        if self.dim_static < 0:
            raise ValueError("dim_static cannot be negative.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_encoder_layers <= 0:
            raise ValueError("num_encoder_layers must be positive.")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads.")
        if self.feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.positional_encoding not in ("sinusoidal", "none"):
            raise ValueError("positional_encoding must be 'sinusoidal' or 'none'.")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")
        if self.residual_sampling not in ("marginal", "joint"):
            raise ValueError("residual_sampling must be 'marginal' or 'joint'.")
        if self.prefix_execution not in ("batched_independent", "serial"):
            raise ValueError(
                "prefix_execution must be 'batched_independent' or 'serial'."
            )
        if self.categorical_loss_weight < 0.0:
            raise ValueError("categorical_loss_weight cannot be negative.")
        if self.continuous_loss_weight < 0.0:
            raise ValueError("continuous_loss_weight cannot be negative.")
        if self.categorical_loss_weight + self.continuous_loss_weight == 0.0:
            raise ValueError("At least one loss weight must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
        if self.scheduler_t0 <= 0 or self.scheduler_t_mult <= 0:
            raise ValueError("Scheduler T_0 and T_mult must be positive.")
        if self.scheduler_eta_min < 0.0:
            raise ValueError("scheduler_eta_min cannot be negative.")

    @property
    def dim_categorical(self) -> int:
        """Width of concatenated one-hot categorical covariates."""

        return sum(self.categorical_cardinalities)

    @property
    def num_categorical_variables(self) -> int:
        return len(self.categorical_cardinalities)


@dataclass(frozen=True)
class StepPrediction:
    """Conditional prediction for the single step after one history prefix."""

    categorical_logits: tuple[Tensor, ...]
    continuous_mean: Tensor

    @property
    def categorical_probabilities(self) -> tuple[Tensor, ...]:
        return tuple(logits.softmax(dim=-1) for logits in self.categorical_logits)


@dataclass(frozen=True)
class ConditionalMeanPrediction:
    """Differentiable, residual-free prediction after one history prefix.

    ``categorical_features`` contains the concatenated marginal class
    probabilities.  ``continuous_mean`` integrates the conditional continuous
    prediction over every joint categorical class assignment, using the same
    independent-head factorization as :meth:`XiongGTransformer.sample_next`.
    """

    categorical_features: Tensor
    categorical_logits: tuple[Tensor, ...]
    continuous_mean: Tensor

    @property
    def categorical_probabilities(self) -> tuple[Tensor, ...]:
        return tuple(logits.softmax(dim=-1) for logits in self.categorical_logits)


@dataclass(frozen=True)
class ConditionalMeanRollout:
    """Autoregressive conditional means under a prescribed action path."""

    categorical_features: Tensor
    categorical_logits: tuple[Tensor, ...]
    continuous_mean: Tensor

    @property
    def categorical_probabilities(self) -> tuple[Tensor, ...]:
        return tuple(logits.softmax(dim=-1) for logits in self.categorical_logits)


@dataclass(frozen=True)
class PrefixPredictions:
    """Last-step predictions from independently evaluated history prefixes."""

    categorical_logits: tuple[Tensor, ...]
    continuous_mean: Tensor

    @property
    def categorical_probabilities(self) -> tuple[Tensor, ...]:
        return tuple(logits.softmax(dim=-1) for logits in self.categorical_logits)


@dataclass(frozen=True)
class GTransformerLoss:
    """Published categorical CE and continuous MSE objectives."""

    total: Tensor
    categorical_cross_entropy: Tensor
    continuous_mse: Tensor
    predictions: PrefixPredictions


@dataclass(frozen=True)
class GTransformerDraw:
    """One Monte Carlo draw of the next covariate vector."""

    categorical_classes: Tensor
    categorical_features: Tensor
    categorical_logits: tuple[Tensor, ...]
    continuous_mean: Tensor
    residual: Tensor
    continuous: Tensor


@dataclass(frozen=True)
class EmpiricalResidualBank:
    """Empirical continuous residuals computed on a holdout split.

    The bank deliberately contains no fitted parametric distribution.  Its
    rows are ``observed - conditional_mean`` residuals.  ``marginal`` sampling
    draws each continuous component independently, matching the variable-wise
    residual draw in the paper; ``joint`` sampling resamples complete rows and
    is exposed as a sensitivity option.
    """

    residuals: Tensor

    def __post_init__(self) -> None:
        if self.residuals.ndim != 2:
            raise ValueError("residuals must have shape [observations, variables].")
        if self.residuals.shape[0] == 0 or self.residuals.shape[1] == 0:
            raise ValueError("residuals must be non-empty.")
        if not self.residuals.is_floating_point():
            raise TypeError("residuals must be floating point.")
        if not torch.isfinite(self.residuals).all():
            raise ValueError("residuals contain non-finite values.")

    @classmethod
    def from_predictions(
        cls,
        observed: Tensor,
        predicted_mean: Tensor,
        valid_mask: Tensor | None = None,
        *,
        storage_device: torch.device | str = "cpu",
    ) -> "EmpiricalResidualBank":
        """Construct a bank from holdout observations and teacher-forced means."""

        if observed.shape != predicted_mean.shape:
            raise ValueError(
                "observed and predicted_mean must have identical shapes, got "
                f"{tuple(observed.shape)} and {tuple(predicted_mean.shape)}."
            )
        if observed.ndim < 2:
            raise ValueError("observed must have at least batch and variable axes.")
        leading_shape = observed.shape[:-1]
        if valid_mask is None:
            selected = (observed - predicted_mean).reshape(-1, observed.shape[-1])
        else:
            if tuple(valid_mask.shape) != tuple(leading_shape):
                raise ValueError(
                    f"valid_mask must have shape {tuple(leading_shape)}, got "
                    f"{tuple(valid_mask.shape)}."
                )
            if valid_mask.dtype is not torch.bool:
                raise TypeError("valid_mask must be boolean.")
            selected = (observed - predicted_mean)[valid_mask]
        return cls(selected.detach().to(device=storage_device).contiguous())

    @property
    def dim_continuous(self) -> int:
        return self.residuals.shape[1]

    def to(self, device: torch.device | str) -> "EmpiricalResidualBank":
        return EmpiricalResidualBank(self.residuals.to(device=device))

    def sample(
        self,
        sample_shape: int | Sequence[int],
        *,
        sampling: ResidualSampling = "marginal",
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Bootstrap residuals without imposing a parametric noise model."""

        if isinstance(sample_shape, int):
            shape = (sample_shape,)
        else:
            shape = tuple(sample_shape)
        if not shape or any(size < 0 for size in shape):
            raise ValueError("sample_shape must contain one or more nonnegative sizes.")
        if sampling not in ("marginal", "joint"):
            raise ValueError("sampling must be 'marginal' or 'joint'.")

        bank = self.residuals
        n_rows, n_variables = bank.shape
        random_device = (
            bank.device
            if generator is None
            else torch.device(getattr(generator, "device", bank.device))
        )
        if sampling == "joint":
            indices = torch.randint(
                n_rows,
                shape,
                generator=generator,
                device=random_device,
            )
            indices = indices.to(device=bank.device)
            draws = bank[indices]
        else:
            indices = torch.randint(
                n_rows,
                (*shape, n_variables),
                generator=generator,
                device=random_device,
            )
            indices = indices.to(device=bank.device)
            columns = torch.arange(n_variables, device=bank.device)
            columns = columns.reshape(*([1] * len(shape)), n_variables)
            draws = bank[indices, columns]

        return draws.to(
            device=bank.device if device is None else device,
            dtype=bank.dtype if dtype is None else dtype,
        )


class _SinusoidalPositionEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_length: int) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_length, hidden_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * scales)
        if hidden_dim > 1:
            encoding[:, 1::2] = torch.cos(positions * scales[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        length = values.shape[1]
        if length > self.encoding.shape[1]:
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum "
                f"{self.encoding.shape[1]}."
            )
        return values + self.encoding[:, :length].to(
            device=values.device, dtype=values.dtype
        )


class XiongGTransformer(nn.Module):
    """Independent implementation of the two-encoder G-Transformer core.

    A prefix contains aligned tokens ``concat(L_t, A_t)``.  No causal mask is
    needed inside an encoder call because every token is observed history.  To
    preserve Xiong et al.'s custom training algorithm, :meth:`predict_prefixes`
    invokes each encoder separately on prefixes of lengths ``1, ..., T`` and
    retains only the final hidden state from each call.
    """

    def __init__(self, config: GTransformerConfig) -> None:
        super().__init__()
        self.config = config
        history_dim = (
            config.dim_categorical
            + config.dim_continuous
            + config.dim_action
            + config.dim_static
        )
        self.categorical_input_projection = nn.Linear(history_dim, config.hidden_dim)
        self.continuous_input_projection = nn.Linear(
            history_dim + config.dim_categorical, config.hidden_dim
        )
        self.categorical_encoder = self._make_encoder(config)
        self.continuous_encoder = self._make_encoder(config)
        self.categorical_heads = nn.ModuleList(
            nn.Linear(config.hidden_dim, cardinality)
            for cardinality in config.categorical_cardinalities
        )
        self.continuous_head = nn.Linear(config.hidden_dim, config.dim_continuous)

        if config.positional_encoding == "sinusoidal":
            self.categorical_position: nn.Module = _SinusoidalPositionEncoding(
                config.hidden_dim, config.max_sequence_length
            )
            self.continuous_position: nn.Module = _SinusoidalPositionEncoding(
                config.hidden_dim, config.max_sequence_length
            )
        else:
            self.categorical_position = nn.Identity()
            self.continuous_position = nn.Identity()

    @staticmethod
    def _make_encoder(config: GTransformerConfig) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        # The batched-independent prefix implementation uses right-padding
        # masks.  PyTorch's nested-tensor mask path is unavailable on MPS and
        # is not needed for these short, fixed-horizon sequences.
        return nn.TransformerEncoder(
            layer,
            num_layers=config.num_encoder_layers,
            enable_nested_tensor=False,
        )

    def _validate_history(
        self,
        categorical_history: Tensor,
        continuous_history: Tensor,
        actions: Tensor,
        static_features: Tensor | None,
        padding_mask: Tensor | None,
    ) -> tuple[int, int]:
        if categorical_history.ndim != 3:
            raise ValueError(
                "categorical_history must have shape [batch, time, one_hot_width]."
            )
        batch, time, categorical_dim = categorical_history.shape
        expected = {
            "categorical_history": (batch, time, self.config.dim_categorical),
            "continuous_history": (batch, time, self.config.dim_continuous),
            "actions": (batch, time, self.config.dim_action),
        }
        actual = {
            "categorical_history": tuple(categorical_history.shape),
            "continuous_history": tuple(continuous_history.shape),
            "actions": tuple(actions.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} must have shape {shape}, got {actual[name]}.")
        if time == 0:
            raise ValueError("History must contain at least one time step.")
        tensors = (categorical_history, continuous_history, actions)
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise TypeError("History and action tensors must be floating point.")
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("History and action tensors must be finite.")
        if any(tensor.device != categorical_history.device for tensor in tensors):
            raise ValueError("History and action tensors must share one device.")

        if self.config.dim_static:
            expected_static = (batch, self.config.dim_static)
            if static_features is None or tuple(static_features.shape) != expected_static:
                actual_static = None if static_features is None else tuple(static_features.shape)
                raise ValueError(
                    f"static_features must have shape {expected_static}, got {actual_static}."
                )
            if not static_features.is_floating_point() or not torch.isfinite(
                static_features
            ).all():
                raise ValueError("static_features must be finite floating point values.")
            if static_features.device != categorical_history.device:
                raise ValueError("static_features must be on the history device.")
        elif static_features is not None and tuple(static_features.shape) != (batch, 0):
            raise ValueError("static_features must be omitted when dim_static is zero.")

        if padding_mask is not None:
            if tuple(padding_mask.shape) != (batch, time):
                raise ValueError(
                    f"padding_mask must have shape {(batch, time)}, got "
                    f"{tuple(padding_mask.shape)}."
                )
            if padding_mask.dtype is not torch.bool:
                raise TypeError("padding_mask must be boolean, with True marking padding.")
            if padding_mask.device != categorical_history.device:
                raise ValueError("padding_mask must be on the history device.")
            if padding_mask.all(dim=1).any():
                raise ValueError("Every history must contain at least one unpadded token.")
            if ((~padding_mask[:, 1:]) & padding_mask[:, :-1]).any():
                raise ValueError("padding_mask must describe right-padded histories.")
        return batch, time

    def _history_features(
        self,
        categorical_history: Tensor,
        continuous_history: Tensor,
        actions: Tensor,
        static_features: Tensor | None,
    ) -> Tensor:
        parts = [categorical_history, continuous_history, actions]
        if self.config.dim_static:
            assert static_features is not None
            repeated_static = static_features.unsqueeze(1).expand(
                -1, categorical_history.shape[1], -1
            )
            parts.append(repeated_static)
        return torch.cat(parts, dim=-1)

    def _normalize_classes(
        self,
        classes: Tensor,
        *,
        batch: int,
        name: str,
    ) -> Tensor:
        if classes.ndim == 1 and self.config.num_categorical_variables == 1:
            classes = classes.unsqueeze(-1)
        expected = (batch, self.config.num_categorical_variables)
        if tuple(classes.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(classes.shape)}.")
        if classes.dtype == torch.bool or classes.is_floating_point():
            raise TypeError(f"{name} must contain integer class indices.")
        for variable, cardinality in enumerate(self.config.categorical_cardinalities):
            values = classes[:, variable]
            if ((values < 0) | (values >= cardinality)).any():
                raise ValueError(
                    f"{name} contains an out-of-range value for categorical variable "
                    f"{variable} with {cardinality} classes."
                )
        return classes.to(dtype=torch.long)

    def encode_categorical_classes(
        self,
        classes: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        """Concatenate one-hot encodings of class-index columns."""

        if classes.ndim != 2:
            raise ValueError("classes must have shape [batch, categorical_variables].")
        encoded = [
            F.one_hot(classes[:, variable], num_classes=cardinality).to(dtype=dtype)
            for variable, cardinality in enumerate(self.config.categorical_cardinalities)
        ]
        return torch.cat(encoded, dim=-1)

    @staticmethod
    def _last_hidden(
        encoder: nn.TransformerEncoder,
        values: Tensor,
        padding_mask: Tensor | None,
    ) -> Tensor:
        encoded = encoder(values, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            return encoded[:, -1]
        last_indices = (~padding_mask).sum(dim=1) - 1
        batch_indices = torch.arange(encoded.shape[0], device=encoded.device)
        return encoded[batch_indices, last_indices]

    def _categorical_prediction(
        self,
        history_features: Tensor,
        padding_mask: Tensor | None,
    ) -> tuple[Tensor, ...]:
        embedded = self.categorical_position(
            self.categorical_input_projection(history_features)
        )
        hidden = self._last_hidden(self.categorical_encoder, embedded, padding_mask)
        logits = tuple(head(hidden) for head in self.categorical_heads)
        if self.config.categorical_head_relu:
            logits = tuple(F.relu(value) for value in logits)
        return logits

    def _continuous_prediction(
        self,
        history_features: Tensor,
        next_categorical_features: Tensor,
        padding_mask: Tensor | None,
    ) -> Tensor:
        repeated_next = next_categorical_features.unsqueeze(1).expand(
            -1, history_features.shape[1], -1
        )
        continuous_input = torch.cat((history_features, repeated_next), dim=-1)
        embedded = self.continuous_position(
            self.continuous_input_projection(continuous_input)
        )
        hidden = self._last_hidden(self.continuous_encoder, embedded, padding_mask)
        return self.continuous_head(hidden)

    def predict_next(
        self,
        categorical_history: Tensor,
        continuous_history: Tensor,
        actions: Tensor,
        next_categorical: Tensor,
        *,
        static_features: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> StepPrediction:
        """Predict ``L_{t+1}`` from one generated or observed history prefix.

        ``next_categorical`` supplies the teacher value during training.  For a
        genuine rollout step, call :meth:`sample_next`, which samples this
        value from the categorical encoder before evaluating the continuous
        encoder.
        """

        batch, _ = self._validate_history(
            categorical_history,
            continuous_history,
            actions,
            static_features,
            padding_mask,
        )
        next_categorical = self._normalize_classes(
            next_categorical, batch=batch, name="next_categorical"
        )
        if next_categorical.device != categorical_history.device:
            raise ValueError("next_categorical must be on the history device.")
        history_features = self._history_features(
            categorical_history, continuous_history, actions, static_features
        )
        categorical_logits = self._categorical_prediction(
            history_features, padding_mask
        )
        categorical_features = self.encode_categorical_classes(
            next_categorical, dtype=history_features.dtype
        )
        continuous_mean = self._continuous_prediction(
            history_features, categorical_features, padding_mask
        )
        return StepPrediction(categorical_logits, continuous_mean)

    def predict_next_conditional_mean(
        self,
        categorical_history: Tensor,
        continuous_history: Tensor,
        actions: Tensor,
        *,
        static_features: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> ConditionalMeanPrediction:
        """Return a differentiable, residual-free next-step conditional mean.

        The continuous encoder is conditioned on a categorical class draw in
        the published model.  A deterministic mean therefore requires the law
        of total expectation rather than passing a soft category directly to
        that encoder.  This method enumerates the joint class assignments,
        evaluates their conditional continuous means in one expanded batch,
        and weights them by the categorical-head probabilities.  It performs
        no categorical or residual sampling and deliberately remains attached
        to autograd.
        """

        batch, time = self._validate_history(
            categorical_history,
            continuous_history,
            actions,
            static_features,
            padding_mask,
        )
        history_features = self._history_features(
            categorical_history, continuous_history, actions, static_features
        )
        categorical_logits = self._categorical_prediction(
            history_features, padding_mask
        )
        categorical_probabilities = tuple(
            logits.softmax(dim=-1) for logits in categorical_logits
        )
        categorical_features = torch.cat(categorical_probabilities, dim=-1)

        class_axes = tuple(
            torch.arange(cardinality, device=history_features.device)
            for cardinality in self.config.categorical_cardinalities
        )
        joint_classes = torch.cartesian_prod(*class_axes).reshape(
            -1, self.config.num_categorical_variables
        )
        combinations = joint_classes.shape[0]
        joint_categorical_features = self.encode_categorical_classes(
            joint_classes,
            dtype=history_features.dtype,
        )
        joint_categorical_features = (
            joint_categorical_features.unsqueeze(0)
            .expand(batch, combinations, self.config.dim_categorical)
            .reshape(batch * combinations, self.config.dim_categorical)
        )
        feature_dim = history_features.shape[-1]
        expanded_history = (
            history_features.unsqueeze(1)
            .expand(batch, combinations, time, feature_dim)
            .reshape(batch * combinations, time, feature_dim)
        )
        expanded_padding_mask = None
        if padding_mask is not None:
            expanded_padding_mask = (
                padding_mask.unsqueeze(1)
                .expand(batch, combinations, time)
                .reshape(batch * combinations, time)
            )
        conditional_means = self._continuous_prediction(
            expanded_history,
            joint_categorical_features,
            expanded_padding_mask,
        ).reshape(batch, combinations, self.config.dim_continuous)

        joint_probability = history_features.new_ones((batch, combinations))
        for variable, probabilities in enumerate(categorical_probabilities):
            joint_probability = joint_probability * probabilities[
                :, joint_classes[:, variable]
            ]
        continuous_mean = torch.sum(
            joint_probability.unsqueeze(-1) * conditional_means,
            dim=1,
        )
        return ConditionalMeanPrediction(
            categorical_features,
            categorical_logits,
            continuous_mean,
        )

    def rollout_prescribed_actions(
        self,
        initial_categorical: Tensor,
        initial_continuous: Tensor,
        prescribed_actions: Tensor,
        *,
        static_features: Tensor | None = None,
    ) -> ConditionalMeanRollout:
        """Autoregress conditional means through a prescribed action path.

        Inputs describe the initial covariate vector and one action for every
        requested transition.  Generated marginal categorical probabilities
        and continuous conditional means are appended to the model history at
        each step.  No residual bank, random generator, detachment, or
        ``no_grad`` context is used, so the complete path remains
        differentiable with respect to the model and its floating-point
        inputs.
        """

        if initial_categorical.ndim != 2:
            raise ValueError(
                "initial_categorical must have shape [batch, one_hot_width]."
            )
        batch = initial_categorical.shape[0]
        expected_categorical = (batch, self.config.dim_categorical)
        expected_continuous = (batch, self.config.dim_continuous)
        if tuple(initial_categorical.shape) != expected_categorical:
            raise ValueError(
                "initial_categorical must have shape "
                f"{expected_categorical}, got {tuple(initial_categorical.shape)}."
            )
        if tuple(initial_continuous.shape) != expected_continuous:
            raise ValueError(
                "initial_continuous must have shape "
                f"{expected_continuous}, got {tuple(initial_continuous.shape)}."
            )
        if prescribed_actions.ndim != 3:
            raise ValueError(
                "prescribed_actions must have shape [batch, horizon, action_width]."
            )
        horizon = prescribed_actions.shape[1]
        expected_actions = (batch, horizon, self.config.dim_action)
        if tuple(prescribed_actions.shape) != expected_actions:
            raise ValueError(
                "prescribed_actions must have shape "
                f"{expected_actions}, got {tuple(prescribed_actions.shape)}."
            )
        if horizon == 0:
            raise ValueError("prescribed_actions must contain at least one step.")

        categorical_history = initial_categorical.unsqueeze(1)
        continuous_history = initial_continuous.unsqueeze(1)
        categorical_by_step: list[Tensor] = []
        categorical_logits_by_step: list[tuple[Tensor, ...]] = []
        continuous_by_step: list[Tensor] = []
        for step in range(horizon):
            prediction = self.predict_next_conditional_mean(
                categorical_history,
                continuous_history,
                prescribed_actions[:, : step + 1],
                static_features=static_features,
            )
            categorical_by_step.append(prediction.categorical_features)
            categorical_logits_by_step.append(prediction.categorical_logits)
            continuous_by_step.append(prediction.continuous_mean)
            categorical_history = torch.cat(
                (categorical_history, prediction.categorical_features.unsqueeze(1)),
                dim=1,
            )
            continuous_history = torch.cat(
                (continuous_history, prediction.continuous_mean.unsqueeze(1)),
                dim=1,
            )

        categorical_logits = tuple(
            torch.stack(
                [step_logits[variable] for step_logits in categorical_logits_by_step],
                dim=1,
            )
            for variable in range(self.config.num_categorical_variables)
        )
        return ConditionalMeanRollout(
            torch.stack(categorical_by_step, dim=1),
            categorical_logits,
            torch.stack(continuous_by_step, dim=1),
        )

    def predict_prefixes(
        self,
        categorical_history: Tensor,
        continuous_history: Tensor,
        actions: Tensor,
        next_categorical: Tensor,
        *,
        static_features: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> PrefixPredictions:
        """Predict every next step using independent prefixes, as in the paper.

        For output time ``t``, the corresponding encoder example can attend
        only to input positions ``0..t`` and only its final unpadded state is
        retained.  ``batched_independent`` expands all prefixes into separate
        right-padded batch examples and evaluates them together; it does not
        expose future tokens and is therefore distinct from sequence-to-
        sequence training.  ``serial`` retains the direct reference loop.
        """

        batch, time = self._validate_history(
            categorical_history,
            continuous_history,
            actions,
            static_features,
            padding_mask,
        )
        if next_categorical.ndim == 2 and self.config.num_categorical_variables == 1:
            next_categorical = next_categorical.unsqueeze(-1)
        expected = (batch, time, self.config.num_categorical_variables)
        if tuple(next_categorical.shape) != expected:
            raise ValueError(
                f"next_categorical must have shape {expected}, got "
                f"{tuple(next_categorical.shape)}."
            )
        if next_categorical.device != categorical_history.device:
            raise ValueError("next_categorical must be on the history device.")
        normalized_classes = self._normalize_classes(
            next_categorical.reshape(
                batch * time, self.config.num_categorical_variables
            ),
            batch=batch * time,
            name="next_categorical",
        ).reshape(batch, time, self.config.num_categorical_variables)

        if self.config.prefix_execution == "serial":
            return self._predict_prefixes_serial(
                categorical_history,
                continuous_history,
                actions,
                normalized_classes,
                static_features=static_features,
                padding_mask=padding_mask,
            )
        return self._predict_prefixes_batched_independent(
            categorical_history,
            continuous_history,
            actions,
            normalized_classes,
            static_features=static_features,
            padding_mask=padding_mask,
        )

    def _predict_prefixes_serial(
        self,
        categorical_history: Tensor,
        continuous_history: Tensor,
        actions: Tensor,
        next_categorical: Tensor,
        *,
        static_features: Tensor | None,
        padding_mask: Tensor | None,
    ) -> PrefixPredictions:
        """Reference implementation with one encoder invocation per prefix."""

        time = categorical_history.shape[1]

        categorical_by_prefix: list[tuple[Tensor, ...]] = []
        continuous_by_prefix: list[Tensor] = []
        for end in range(1, time + 1):
            step = self.predict_next(
                categorical_history[:, :end],
                continuous_history[:, :end],
                actions[:, :end],
                next_categorical[:, end - 1],
                static_features=static_features,
                padding_mask=None if padding_mask is None else padding_mask[:, :end],
            )
            categorical_by_prefix.append(step.categorical_logits)
            continuous_by_prefix.append(step.continuous_mean)

        categorical_logits = tuple(
            torch.stack(
                [step_logits[variable] for step_logits in categorical_by_prefix],
                dim=1,
            )
            for variable in range(self.config.num_categorical_variables)
        )
        continuous_mean = torch.stack(continuous_by_prefix, dim=1)
        return PrefixPredictions(categorical_logits, continuous_mean)

    def _predict_prefixes_batched_independent(
        self,
        categorical_history: Tensor,
        continuous_history: Tensor,
        actions: Tensor,
        next_categorical: Tensor,
        *,
        static_features: Tensor | None,
        padding_mask: Tensor | None,
    ) -> PrefixPredictions:
        """Evaluate every independently masked prefix in two encoder calls.

        Batch element ``(patient, end)`` contains the patient's complete input
        storage but masks every position at or after ``end``.  Transformer
        layers do not mix batch elements, and only position ``end - 1`` (or
        the last earlier observed position under external padding) is gathered.
        Thus no prediction can attend to a later token from its source patient.
        """

        batch, time, _ = categorical_history.shape
        history_features = self._history_features(
            categorical_history,
            continuous_history,
            actions,
            static_features,
        )
        feature_dim = history_features.shape[-1]
        expanded_history = (
            history_features.unsqueeze(1)
            .expand(batch, time, time, feature_dim)
            .reshape(batch * time, time, feature_dim)
        )

        prefix_lengths = torch.arange(
            1, time + 1, device=history_features.device
        ).repeat(batch)
        positions = torch.arange(time, device=history_features.device).unsqueeze(0)
        expanded_padding_mask = positions >= prefix_lengths.unsqueeze(1)
        if padding_mask is not None:
            observed_padding = (
                padding_mask.unsqueeze(1)
                .expand(batch, time, time)
                .reshape(batch * time, time)
            )
            expanded_padding_mask = expanded_padding_mask | observed_padding

        categorical_logits_flat = self._categorical_prediction(
            expanded_history,
            expanded_padding_mask,
        )
        next_categorical_features = self.encode_categorical_classes(
            next_categorical.reshape(
                batch * time, self.config.num_categorical_variables
            ),
            dtype=history_features.dtype,
        )
        continuous_mean_flat = self._continuous_prediction(
            expanded_history,
            next_categorical_features,
            expanded_padding_mask,
        )
        categorical_logits = tuple(
            logits.reshape(batch, time, cardinality)
            for logits, cardinality in zip(
                categorical_logits_flat,
                self.config.categorical_cardinalities,
                strict=True,
            )
        )
        continuous_mean = continuous_mean_flat.reshape(
            batch, time, self.config.dim_continuous
        )
        return PrefixPredictions(categorical_logits, continuous_mean)

    def forward(
        self,
        categorical_history: Tensor,
        continuous_history: Tensor,
        actions: Tensor,
        next_categorical: Tensor,
        *,
        static_features: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> PrefixPredictions:
        return self.predict_prefixes(
            categorical_history,
            continuous_history,
            actions,
            next_categorical,
            static_features=static_features,
            padding_mask=padding_mask,
        )

    def training_loss(
        self,
        categorical_history: Tensor,
        continuous_history: Tensor,
        actions: Tensor,
        next_categorical: Tensor,
        next_continuous: Tensor,
        *,
        static_features: Tensor | None = None,
        padding_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> GTransformerLoss:
        """Compute mean categorical CE plus mean continuous MSE."""

        predictions = self.predict_prefixes(
            categorical_history,
            continuous_history,
            actions,
            next_categorical,
            static_features=static_features,
            padding_mask=padding_mask,
        )
        batch, time, _ = predictions.continuous_mean.shape
        if next_categorical.ndim == 2 and self.config.num_categorical_variables == 1:
            next_categorical = next_categorical.unsqueeze(-1)
        expected_categorical = (batch, time, self.config.num_categorical_variables)
        expected_continuous = (batch, time, self.config.dim_continuous)
        if tuple(next_categorical.shape) != expected_categorical:
            raise ValueError(
                f"next_categorical must have shape {expected_categorical}, got "
                f"{tuple(next_categorical.shape)}."
            )
        if tuple(next_continuous.shape) != expected_continuous:
            raise ValueError(
                f"next_continuous must have shape {expected_continuous}, got "
                f"{tuple(next_continuous.shape)}."
            )
        if next_continuous.device != predictions.continuous_mean.device:
            raise ValueError("next_continuous must be on the history device.")
        if not next_continuous.is_floating_point() or not torch.isfinite(
            next_continuous
        ).all():
            raise ValueError("next_continuous must be finite floating point values.")
        if target_mask is None:
            target_mask = torch.ones(
                batch, time, dtype=torch.bool, device=predictions.continuous_mean.device
            )
            if padding_mask is not None:
                target_mask = target_mask & ~padding_mask
        else:
            if tuple(target_mask.shape) != (batch, time):
                raise ValueError(
                    f"target_mask must have shape {(batch, time)}, got "
                    f"{tuple(target_mask.shape)}."
                )
            if target_mask.dtype is not torch.bool:
                raise TypeError("target_mask must be boolean.")
            if target_mask.device != predictions.continuous_mean.device:
                raise ValueError("target_mask must be on the history device.")
        if not target_mask.any():
            raise ValueError("target_mask selects no targets.")

        categorical_losses = []
        targets = next_categorical.to(dtype=torch.long)
        for variable, logits in enumerate(predictions.categorical_logits):
            categorical_losses.append(
                F.cross_entropy(
                    logits[target_mask],
                    targets[..., variable][target_mask],
                    reduction="mean",
                )
            )
        categorical_cross_entropy = torch.stack(categorical_losses).mean()
        continuous_mse = F.mse_loss(
            predictions.continuous_mean[target_mask],
            next_continuous[target_mask],
            reduction="mean",
        )
        total = (
            self.config.categorical_loss_weight * categorical_cross_entropy
            + self.config.continuous_loss_weight * continuous_mse
        )
        return GTransformerLoss(
            total,
            categorical_cross_entropy,
            continuous_mse,
            predictions,
        )

    @torch.no_grad()
    def residual_bank_from_holdout(
        self,
        categorical_history: Tensor,
        continuous_history: Tensor,
        actions: Tensor,
        next_categorical: Tensor,
        next_continuous: Tensor,
        *,
        static_features: Tensor | None = None,
        padding_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        storage_device: torch.device | str = "cpu",
    ) -> EmpiricalResidualBank:
        """Estimate the nonparametric simulation noise on holdout data."""

        was_training = self.training
        self.eval()
        try:
            predictions = self.predict_prefixes(
                categorical_history,
                continuous_history,
                actions,
                next_categorical,
                static_features=static_features,
                padding_mask=padding_mask,
            )
        finally:
            self.train(was_training)
        if target_mask is None and padding_mask is not None:
            target_mask = ~padding_mask
        return EmpiricalResidualBank.from_predictions(
            next_continuous,
            predictions.continuous_mean,
            target_mask,
            storage_device=storage_device,
        )

    @torch.no_grad()
    def sample_next(
        self,
        categorical_history: Tensor,
        continuous_history: Tensor,
        actions: Tensor,
        residual_bank: EmpiricalResidualBank,
        *,
        static_features: Tensor | None = None,
        padding_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
        residual_sampling: ResidualSampling | None = None,
    ) -> GTransformerDraw:
        """Perform one categorical-first Monte Carlo g-computation step."""

        batch, _ = self._validate_history(
            categorical_history,
            continuous_history,
            actions,
            static_features,
            padding_mask,
        )
        if residual_bank.dim_continuous != self.config.dim_continuous:
            raise ValueError(
                "Residual bank width does not match dim_continuous: "
                f"{residual_bank.dim_continuous} != {self.config.dim_continuous}."
            )
        history_features = self._history_features(
            categorical_history, continuous_history, actions, static_features
        )
        categorical_logits = self._categorical_prediction(
            history_features, padding_mask
        )
        sampled_classes = torch.stack(
            [
                torch.multinomial(
                    logits.softmax(dim=-1).to(
                        device=(
                            logits.device
                            if generator is None
                            else torch.device(
                                getattr(generator, "device", logits.device)
                            )
                        )
                    ),
                    num_samples=1,
                    generator=generator,
                )
                .squeeze(-1)
                .to(device=logits.device)
                for logits in categorical_logits
            ],
            dim=-1,
        )
        categorical_features = self.encode_categorical_classes(
            sampled_classes, dtype=history_features.dtype
        )
        continuous_mean = self._continuous_prediction(
            history_features, categorical_features, padding_mask
        )
        residual = residual_bank.sample(
            batch,
            sampling=(
                self.config.residual_sampling
                if residual_sampling is None
                else residual_sampling
            ),
            generator=generator,
            device=continuous_mean.device,
            dtype=continuous_mean.dtype,
        )
        continuous = continuous_mean + residual
        return GTransformerDraw(
            sampled_classes,
            categorical_features,
            categorical_logits,
            continuous_mean,
            residual,
            continuous,
        )


def build_optimizer_and_scheduler(
    model: XiongGTransformer,
) -> tuple[
    torch.optim.Adam,
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
]:
    """Build the Adam/scheduler pair reported for the paper's CVSim study."""

    config = model.config
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.scheduler_t0,
        T_mult=config.scheduler_t_mult,
        eta_min=config.scheduler_eta_min,
    )
    return optimizer, scheduler
