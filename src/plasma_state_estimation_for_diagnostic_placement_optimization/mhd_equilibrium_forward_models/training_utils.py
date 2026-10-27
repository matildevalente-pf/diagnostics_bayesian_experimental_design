# The training loop of the MLP surrogate: a curriculum of stages, each a jitted gradient
# step under the stage's loss terms with early stopping on the validation rows.

import csv
import functools
import logging
import pathlib
import time
from collections.abc import Callable
from typing import NamedTuple, cast

import flax.typing
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Float
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_types,
    mlp_types,
    mlp_utils,
    physics_loss_utils,
    preprocessing_types,
    preprocessing_utils,
)

logger = logging.getLogger(__name__)

MAX_LEARNING_RATE_REDUCTIONS = 3
"""Times a stage divides its learning rate by ten on a plateau before it stops."""


class EpochRecord(NamedTuple):
    """One epoch's losses: `train_` terms are batch means with dropout on, `validation_`
    terms one dropout-off evaluation of the parameters the epoch ended at."""

    stage: str
    epoch: int
    learning_rate: float
    train_data: float
    train_fnorm: float
    train_total: float
    validation_data: float
    validation_fnorm: float
    validation_total: float


class StageResult(NamedTuple):
    """What one curriculum stage produced."""

    params: flax.typing.VariableDict
    """Parameters of the selected epoch."""

    selected: EpochRecord
    """The epoch that minimized the stage's `selection_term` on the validation rows."""

    selection_term: mlp_types.SelectionTerm

    history: list[EpochRecord]
    """One record per epoch run."""

    fnorm_weight: float | None
    """Weight the force-residual term ran at, after any calibration; None without it."""

    @property
    def n_epochs(self) -> int:
        return len(self.history)


class TrainingResult(NamedTuple):
    """What a whole curriculum produced: a trained model and the run behind it."""

    params: flax.typing.VariableDict
    """Parameters of the last stage's selected epoch."""

    preprocessor: preprocessing_types.PreprocessorParams
    """The normalization fitted on the training rows, part of the model."""

    stages: list[StageResult]

    train_seconds: float

    force_normalization_constant: float | None
    """Constant the force-residual term divided by; it is set by the template row, so it
    changes with the dataset, the filter and the split."""

    fnorm_template_equilibrium_id: str | None
    """Id of the row the force-residual term was built on."""

    @property
    def history(self) -> list[EpochRecord]:
        return [record for stage in self.stages for record in stage.history]

    @property
    def train_mse(self) -> float:
        return self.stages[-1].selected.train_data

    @property
    def validation_total(self) -> float:
        return self.stages[-1].selected.validation_total

    @property
    def n_epochs(self) -> int:
        return sum(stage.n_epochs for stage in self.stages)


class Rows(NamedTuple):
    """A set of rows in every form the loss terms need; a pytree, so jit can take it."""

    x: Float[jax.Array, "n_rows n_rho"]
    """Normalized pressure profiles, the network's input."""

    y_latent: Float[jax.Array, "n_rows n_latent"]
    """Targets in the network's own space."""

    fnorm_target: Float[jax.Array, "n_rows n_y_and_knots"]
    """Flat target vectors with the pressure at the spline knots [Pa] appended, for the
    force-residual term."""

    def take(self, rows: NpOrJaxArray | slice) -> "Rows":
        return Rows(*(array[rows] for array in self))


class EarlyStopping:
    """Model selection and the learning-rate ladder of one stage.

    `patience` epochs without improvement, once past `min_epochs`, divide the learning
    rate by ten; after `MAX_LEARNING_RATE_REDUCTIONS` of those the stage is `exhausted`.
    """

    def __init__(self, stage: mlp_types.CurriculumStepConfig) -> None:
        self._stage = stage
        self.learning_rate = float(stage.learning_rate)
        self.best = float("inf")
        self.exhausted = False
        self._epochs_without_improvement = 0
        self._n_reductions = 0

    def update(self, epoch: int, criterion: float) -> bool:
        """Record one epoch's criterion; True when it is the best so far."""
        stage = self._stage
        if criterion < self.best:
            self.best = criterion
            self._epochs_without_improvement = 0
            return True
        self._epochs_without_improvement += 1
        plateaued = (
            stage.patience is not None
            and epoch >= stage.min_epochs
            and self._epochs_without_improvement >= stage.patience
        )
        if not plateaued:
            return False
        if self._n_reductions == MAX_LEARNING_RATE_REDUCTIONS:
            self.exhausted = True
            logger.info(
                "[%s] stopping at epoch %d, %d learning-rate reductions spent.",
                stage.label,
                epoch,
                self._n_reductions,
            )
            return False
        self._n_reductions += 1
        self.learning_rate /= 10.0
        self._epochs_without_improvement = 0
        logger.info(
            "[%s] learning-rate reduction %d of %d: %.2e at epoch %d.",
            stage.label,
            self._n_reductions,
            MAX_LEARNING_RATE_REDUCTIONS,
            self.learning_rate,
            epoch,
        )
        return False


def train_mlp_curriculum(
    curriculum: mlp_types.CurriculumLearningConfig,
    settings: mlp_types.MLPSettings,
    dataset: dataset_types.RawDataset,
    split: dataset_types.DataSplit,
) -> TrainingResult:
    """Train one MLP surrogate through every stage of a curriculum.

    The normalization is fitted on the training rows. The force-residual term, which
    pulls DESC in, is built on the first training row and only if a stage asks for it.
    """
    basis = dataset.basis
    train_rows = np.asarray(split.train)
    validation_rows = np.asarray(split.validation)
    pressure = np.asarray(dataset.pressure, dtype=np.float32)
    y = np.asarray(dataset.y, dtype=np.float32)
    pressure_knots = np.asarray(dataset.pressure_knots)
    if not np.all(pressure_knots == pressure_knots[0]):
        raise ValueError("The force-residual term needs one knot grid for every row.")
    pressure_knot_values = np.asarray(dataset.pressure_params, dtype=np.float32)

    preprocessor = preprocessing_utils.fit_preprocessor(
        pressure[train_rows],
        y[train_rows],
        basis,
        n_pca_components=settings.n_pca_components,
        energy_scale=settings.energy_scale,
    )
    decode = functools.partial(preprocessing_utils.inverse_transform_y, preprocessor)
    train = build_rows(
        preprocessor,
        pressure[train_rows],
        y[train_rows],
        pressure_knot_values[train_rows],
    )
    validation = build_rows(
        preprocessor,
        pressure[validation_rows],
        y[validation_rows],
        pressure_knot_values[validation_rows],
    )

    mlp = mlp_utils.build_mlp(settings.architecture, n_outputs=preprocessor.n_latent)
    init_key, permutation_key = jax.random.split(jax.random.PRNGKey(settings.seed))
    params = mlp_utils.init_params(mlp, basis.n_rho, init_key)
    logger.info(
        "Training %d rows, %d validation rows, %d inputs, %d outputs, %s hidden.",
        len(train_rows),
        len(validation_rows),
        basis.n_rho,
        preprocessor.n_latent,
        settings.architecture.hidden_layer_widths,
    )

    # Unweighted: the stage applies its own weight.
    fnorm_loss = None
    if any(stage.loss_config.use_fnorm for stage in curriculum.stages):
        fnorm_loss = physics_loss_utils.build_fnorm_loss(
            basis,
            np.asarray(dataset.y)[train_rows[0]],
            np.asarray(dataset.pressure_params)[train_rows[0]],
            pressure_knots[0],
            loss_weight=1.0,
        )

    started = time.monotonic()
    stages: list[StageResult] = []
    for index, stage in enumerate(curriculum.stages):
        logger.info(
            "Stage %d of %d: %s", index + 1, len(curriculum.stages), stage.label
        )
        result = run_stage(
            stage,
            settings,
            mlp,
            params,
            train,
            validation,
            decode,
            fnorm_loss,
            permutation_key,
            epoch_offset=sum(done.n_epochs for done in stages),
        )
        params = result.params
        permutation_key, _ = jax.random.split(permutation_key)
        stages.append(result)

    return TrainingResult(
        params=params,
        preprocessor=preprocessor,
        stages=stages,
        train_seconds=time.monotonic() - started,
        force_normalization_constant=(
            None
            if fnorm_loss is None
            else float(fnorm_loss.keywords["normalization_constant"])
        ),
        fnorm_template_equilibrium_id=(
            None if fnorm_loss is None else dataset.equilibrium_ids[int(train_rows[0])]
        ),
    )


def run_stage(
    stage: mlp_types.CurriculumStepConfig,
    settings: mlp_types.MLPSettings,
    mlp: mlp_utils.MLP,
    params: flax.typing.VariableDict,
    train: Rows,
    validation: Rows,
    decode: Callable,
    fnorm_loss: Callable | None,
    permutation_key: jax.Array,
    epoch_offset: int,
) -> StageResult:
    """Run one stage from the parameters the previous one reached.

    Selects the epoch that minimizes the stage's `selection_term` on the validation rows,
    which `EarlyStopping` also reads for its learning-rate ladder.
    """
    use_data, use_fnorm = stage.loss_config.use_data, stage.loss_config.use_fnorm
    fnorm_weight = None
    if use_fnorm:
        if fnorm_loss is None:
            raise ValueError(f"Stage {stage.label!r} uses fnorm but was given no term.")
        fnorm_weight = stage.loss_config.loss_fnorm_weight
        if fnorm_weight is None:
            fnorm_weight = calibrate_fnorm_weight(
                mlp,
                params,
                decode,
                fnorm_loss,
                train.take(slice(0, settings.batch_size)),
                use_data,
            )
            logger.info("[%s] calibrated fnorm weight %.4e", stage.label, fnorm_weight)

    # Bound once per stage: it is a static argument of the jitted step, hashed by
    # identity so rebinding it per epoch would recompile the step every epoch.
    loss = functools.partial(
        loss_terms,
        mlp=mlp,
        use_data=use_data,
        use_fnorm=use_fnorm,
        decode=decode,
        fnorm_loss=fnorm_loss,
    )
    optimizer = optax.inject_hyperparams(optax.adamw)(
        learning_rate=stage.learning_rate, weight_decay=settings.weight_decay
    )
    opt_state = optimizer.init(params)
    dropout_key = jax.random.PRNGKey(settings.seed + epoch_offset)
    stopping = EarlyStopping(stage)
    n_train = train.x.shape[0]

    best_params = params
    selected: EpochRecord | None = None
    history: list[EpochRecord] = []
    for epoch in range(stage.n_epochs):
        permutation_key, subkey = jax.random.split(permutation_key)
        permutation = jax.random.permutation(subkey, n_train)
        batch_terms = []
        for start in range(0, n_train, settings.batch_size):
            dropout_key, step_key = jax.random.split(dropout_key)
            params, opt_state, terms = train_step(
                loss,
                optimizer,
                params,
                opt_state,
                train.take(permutation[start : start + settings.batch_size]),
                fnorm_weight,
                step_key,
            )
            batch_terms.append([float(term) for term in terms])
        train_data, train_fnorm = np.mean(batch_terms, axis=0).tolist()
        validation_data, validation_fnorm = (
            float(term) for term in evaluate(loss, params, validation, fnorm_weight)
        )
        record = EpochRecord(
            stage=stage.label,
            epoch=epoch_offset + epoch,
            learning_rate=stopping.learning_rate,
            train_data=train_data,
            train_fnorm=train_fnorm,
            train_total=train_data + train_fnorm,
            validation_data=validation_data,
            validation_fnorm=validation_fnorm,
            validation_total=validation_data + validation_fnorm,
        )
        history.append(record)
        if epoch % settings.log_every_n_epochs == 0:
            logger.info(
                "[%s] epoch %4d | train %.4e (data %.4e fnorm %.4e) "
                "| validation %.4e (data %.4e fnorm %.4e)",
                stage.label,
                epoch,
                record.train_total,
                train_data,
                train_fnorm,
                record.validation_total,
                validation_data,
                validation_fnorm,
            )

        criterion = {
            "total": record.validation_total,
            "data": record.validation_data,
            "fnorm": record.validation_fnorm,
        }[stage.selection_term]
        if stopping.update(epoch, criterion):
            best_params, selected = params, record
        # Only the rate changes; the optimizer keeps its moments across a reduction.
        opt_state = optax.tree_utils.tree_set(
            opt_state, learning_rate=stopping.learning_rate
        )
        if stopping.exhausted:
            break

    if selected is None:
        raise ValueError(f"Stage {stage.label!r} ran no epochs.")
    return StageResult(
        params=best_params,
        selected=selected,
        selection_term=stage.selection_term,
        history=history,
        fnorm_weight=fnorm_weight,
    )


@functools.partial(jax.jit, static_argnames=("loss", "optimizer"))
def train_step(
    loss: Callable,
    optimizer: optax.GradientTransformation,
    params: flax.typing.VariableDict,
    opt_state: optax.OptState,
    batch: Rows,
    fnorm_weight: float | None,
    step_key: jax.Array,
) -> tuple[flax.typing.VariableDict, optax.OptState, tuple[jax.Array, jax.Array]]:
    """One gradient step on a mini-batch; `loss` is `loss_terms` with its static
    arguments bound. Returns the new parameters and state, and the two loss terms."""
    (_, terms), gradients = jax.value_and_grad(loss, has_aux=True)(
        params, batch, fnorm_weight, step_key
    )
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    # optax types parameter trees loosely
    new_params = cast(flax.typing.VariableDict, optax.apply_updates(params, updates))
    return new_params, opt_state, terms


@functools.partial(jax.jit, static_argnames=("loss",))
def evaluate(
    loss: Callable,
    params: flax.typing.VariableDict,
    rows: Rows,
    fnorm_weight: float | None,
) -> tuple[jax.Array, jax.Array]:
    """Both loss terms with dropout off, over every given row at once."""
    return loss(params, rows, fnorm_weight, None)[1]


def loss_terms(
    params: flax.typing.VariableDict,
    rows: Rows,
    fnorm_weight: float | None,
    step_key: jax.Array | None,
    *,
    mlp: mlp_utils.MLP,
    use_data: bool,
    use_fnorm: bool,
    decode: Callable,
    fnorm_loss: Callable | None,
) -> tuple[Float[jax.Array, ""], tuple[Float[jax.Array, ""], Float[jax.Array, ""]]]:
    """The stage's objective on some rows, and its data and force-residual terms.

    A term the stage leaves out is absent from the compiled graph, not multiplied by
    zero. Dropout is on when a `step_key` is given, so this serves the gradient step
    and the validation alike.
    """
    # Cast because `apply` is typed for its mutable-collections case, a tuple, as well.
    latent = cast(
        jax.Array,
        mlp.apply(params, rows.x, training=False)
        if step_key is None
        else mlp.apply(params, rows.x, training=True, rngs={"dropout": step_key}),
    )
    zero = jnp.zeros((), dtype=jnp.float32)
    data = jnp.mean((latent - rows.y_latent) ** 2) if use_data else zero
    fnorm = zero
    if use_fnorm:
        assert fnorm_loss is not None and fnorm_weight is not None
        fnorm = fnorm_weight * jnp.mean(
            jax.vmap(fnorm_loss)(decode(latent), rows.fnorm_target)
        )
    return data + fnorm, (data, fnorm)


def calibrate_fnorm_weight(
    mlp: mlp_utils.MLP,
    params: flax.typing.VariableDict,
    decode: Callable,
    fnorm_loss: Callable,
    rows: Rows,
    use_data: bool,
) -> float:
    """Weight that gives the force-residual term the data term's magnitude at the
    stage's start, or 0.1 when the data term is off; measured dropout-off on one
    batch. The term is a logarithm of a residual below one, so it keeps its own sign."""
    latent = cast(jax.Array, mlp.apply(params, rows.x, training=False))
    unweighted = abs(
        float(jnp.mean(jax.vmap(fnorm_loss)(decode(latent), rows.fnorm_target)))
    )
    if unweighted == 0.0:
        return 1.0
    target = float(jnp.mean((latent - rows.y_latent) ** 2)) if use_data else 0.1
    return target / unweighted


def build_rows(
    preprocessor: preprocessing_types.PreprocessorParams,
    pressure: Float[NpOrJaxArray, "n_rows n_rho"],
    y: Float[NpOrJaxArray, "n_rows n_y"],
    pressure_knot_values: Float[NpOrJaxArray, "n_rows n_knots"],
) -> Rows:
    """Normalize a set of rows once, ahead of training."""
    return Rows(
        x=jnp.asarray(preprocessing_utils.transform_x(preprocessor, pressure)),
        y_latent=jnp.asarray(preprocessing_utils.transform_y(preprocessor, y)),
        fnorm_target=jnp.concatenate(
            [
                jnp.asarray(y, dtype=jnp.float32),
                jnp.asarray(pressure_knot_values, dtype=jnp.float32),
            ],
            axis=-1,
        ),
    )


def cap_curriculum_epochs(
    curriculum: mlp_types.CurriculumLearningConfig, max_epochs: int | None
) -> mlp_types.CurriculumLearningConfig:
    """Cap every stage's epochs for a cheaper run, scaling `min_epochs` and `patience`
    by the same factor so early stopping stays reachable inside the smaller budget."""
    if max_epochs is None:
        return curriculum
    stages = []
    for stage in curriculum.stages:
        if max_epochs >= stage.n_epochs:
            stages.append(stage)
            continue
        ratio = max_epochs / stage.n_epochs
        update = {"n_epochs": max_epochs, "min_epochs": round(stage.min_epochs * ratio)}
        if stage.patience is not None:
            update["patience"] = max(1, round(stage.patience * ratio))
        stages.append(stage.model_copy(update=update))
    return mlp_types.CurriculumLearningConfig(stages=stages)


def write_history_csv(path: pathlib.Path, history: list[EpochRecord]) -> None:
    """Write a training history out as one row per epoch."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(EpochRecord._fields)
        writer.writerows(history)
