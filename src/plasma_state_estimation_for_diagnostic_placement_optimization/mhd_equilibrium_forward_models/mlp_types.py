# The configuration an MLP surrogate is trained under: the shape of the network, the
# settings every curriculum stage shares, and the per-stage schedule of loss terms.

from typing import Annotated, Literal, Self

import pydantic

ActivationName = Literal["silu", "relu", "gelu", "tanh", "leaky_relu"]
"""The activations both the training forward pass and the ONNX export can express."""

SelectionTerm = Literal["total", "data", "fnorm"]
"""Validation term a stage selects on, named the way `EpochRecord` names its columns."""


class MLPArchitecture(pydantic.BaseModel):
    """The shape of the network: what it takes to rebuild its forward pass.

    Exactly the part of the configuration a hyperparameter search varies. The output
    width is not here: it follows from the fitted preprocessor.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    hidden_layer_widths: list[Annotated[int, pydantic.Field(ge=1)]] = [512, 512, 512]
    """Width of each hidden layer, in order."""

    activation: ActivationName = "leaky_relu"
    """Activation applied after every hidden layer, but not to the output layer."""

    dropout_rate: Annotated[float, pydantic.Field(ge=0.0, lt=1.0)] = 0.0
    """Dropout probability after every hidden layer; zero removes dropout."""


class MLPSettings(pydantic.BaseModel):
    """What a training run holds fixed across every stage of its curriculum.

    The per-stage schedule, meaning the epoch counts, the patience and the learning
    rate, lives in `CurriculumStepConfig` instead.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    architecture: MLPArchitecture = pydantic.Field(default_factory=MLPArchitecture)
    """Shape of the network."""

    weight_decay: Annotated[float, pydantic.Field(ge=0.0)] = 0.0
    """AdamW penalty on the weights; zero selects plain Adam instead."""

    batch_size: Annotated[int, pydantic.Field(ge=1)] = 32
    """Number of samples per gradient step."""

    seed: int = 42
    """Seed of the parameter initialization, the batch permutation and the dropout."""

    log_every_n_epochs: Annotated[int, pydantic.Field(ge=1)] = 50
    """How often a stage logs its running losses."""

    n_pca_components: Annotated[int, pydantic.Field(ge=1)] | None = None
    """Number of PCA components the target is compressed to, or None for no compression.

    A count, never an explained-variance fraction: asking sklearn for a fraction makes
    the width depend on the last bits of a float32 SVD, which is not reproducible.
    """

    energy_scale: Annotated[float, pydantic.Field(gt=0.0)] = 1.5
    """Decay length of the spectral weights the target is divided by before scaling."""


class FinetuneLossConfig(pydantic.BaseModel):
    """Which loss terms one curriculum stage runs with, and how they are weighted.

    The two `use_` flags are static under `jax.jit`, so a disabled term is absent from
    the compiled graph rather than multiplied by zero.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    use_data: bool = True
    """Enable the mean squared error against the target, in the network's own space."""

    use_fnorm: bool = False
    """Enable the force-residual term, which evaluates DESC inside the training step."""

    loss_fnorm_weight: float | None = None
    """Weight of the force-residual term.

    None calibrates it at the start of the stage, to the data term's magnitude or to
    `training_utils.FNORM_ONLY_TARGET` when there is no data term.
    """

    @pydantic.model_validator(mode="after")
    def _check_at_least_one_term_is_enabled(self) -> Self:
        if not (self.use_data or self.use_fnorm):
            raise ValueError("A stage needs at least one of use_data and use_fnorm.")
        return self


class CurriculumStepConfig(pydantic.BaseModel):
    """One stage of a curriculum: its loss terms and its own training schedule.

    Stages run in order, each starting from the best parameters the previous one
    reached.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    loss_config: FinetuneLossConfig
    """Which loss terms this stage runs with."""

    n_epochs: Annotated[int, pydantic.Field(ge=0)]
    """Maximum number of epochs; the stage can still stop earlier."""

    patience: Annotated[int, pydantic.Field(ge=1)] | None
    """Epochs without a validation improvement before the learning rate is reduced.

    After three reductions the stage stops instead. None disables both.
    """

    min_epochs: Annotated[int, pydantic.Field(ge=0)]
    """Epochs to run before patience is allowed to act."""

    learning_rate: Annotated[float, pydantic.Field(gt=0.0)]
    """Learning rate this stage starts at."""

    label: str = ""
    """Short name of the stage, used in its log lines and in the training history."""

    selection_term: SelectionTerm = "total"
    """Which validation term chooses this stage's model, and drives its plateau counter.

    A stage whose point is one physics term can select on that term alone, rather than
    on the `"total"` objective that stops as soon as a rising data term drags it up.
    """

    @pydantic.model_validator(mode="after")
    def _check_the_selected_term_is_active(self) -> Self:
        active = {
            "total": True,
            "data": self.loss_config.use_data,
            "fnorm": self.loss_config.use_fnorm,
        }
        if not active[self.selection_term]:
            raise ValueError(
                f"Stage {self.label!r} selects on {self.selection_term!r}, a term it "
                f"does not run: use_data={self.loss_config.use_data}, "
                f"use_fnorm={self.loss_config.use_fnorm}."
            )
        return self


class CurriculumLearningConfig(pydantic.BaseModel):
    """The ordered stages one model is trained through."""

    model_config = pydantic.ConfigDict(frozen=True)

    stages: Annotated[list[CurriculumStepConfig], pydantic.Field(min_length=1)]
    """The stages, in the order they run."""
