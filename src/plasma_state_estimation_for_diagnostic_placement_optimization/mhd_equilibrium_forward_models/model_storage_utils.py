# Storing a trained surrogate as two files: an `npz` of every array and a `json` of
# everything else, both readable without this package.

import functools
import json
import pathlib
from collections.abc import Callable
from typing import NamedTuple

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    interpolator_utils,
    mlp_types,
    mlp_utils,
    preprocessing_types,
    surrogate_types,
    surrogate_utils,
    training_utils,
)

MLP_MODEL_KIND = "mlp"
"""Value of the `kind` field of a stored MLP surrogate."""

INTERPOLATOR_MODEL_KIND = "interpolator"
"""Value of the `kind` field of a stored equilibrium interpolator."""

_PARAMS_PREFIX = "params/"
_PREPROCESSOR_PREFIX = "preprocessor/"
_BASIS_PREFIX = "basis/"


class LoadedSurrogate(NamedTuple):
    """A stored model of either kind, reduced to what scoring or plotting it needs."""

    name: str
    """Name it is stored under."""

    kind: str
    """`MLP_MODEL_KIND` or `INTERPOLATOR_MODEL_KIND`."""

    basis: surrogate_types.EquilibriumBasisSpecification
    """The basis its output vector is expressed in."""

    predict: Callable
    """One pressure profile in Pascals to one flat target vector, traceable; an MLP
    is evaluated in float64."""

    n_parameters: int
    """Numbers the model stores: the network's weights, or the whole solved lattice."""

    metadata: dict
    """Its stored `json`."""

    machine_scale: surrogate_types.MachineScale
    """The machine `predict` is evaluated at; unit factors for the stored model's own
    normalized units."""


class LoadedMLPModel(NamedTuple):
    """A stored MLP surrogate, read back into the objects its forward pass needs."""

    mlp: mlp_utils.MLP
    """The network, rebuilt from the stored architecture."""

    params: dict
    """Its parameters, in float32."""

    preprocessor: preprocessing_types.PreprocessorParams
    """The normalization it was trained under."""

    basis: surrogate_types.EquilibriumBasisSpecification
    """The basis its output vector is expressed in."""

    metadata: dict
    """The whole stored `json`: the settings, the curriculum and the metrics."""


def save_mlp_model(
    directory: pathlib.Path,
    name: str,
    configuration: str,
    description: str,
    settings: mlp_types.MLPSettings,
    curriculum: mlp_types.CurriculumLearningConfig,
    basis: surrogate_types.EquilibriumBasisSpecification,
    result: training_utils.TrainingResult,
    dataset_name: str,
    run_directory: str = "",
    generated: str = "",
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write a trained MLP surrogate out as an `npz` and a `json`.

    The file stem is the caller's, and `train_surrogate.py` builds it from the run's
    timestamp, so a model never overwrites the one before it and pairs by name with
    its own run directory. That leaves several runs of one configuration side by side,
    which is why each carries the configuration it came from and a description of what
    made it different.

    Args:
        directory: Directory to write the two files into.
        name: File stem of both files, unique across runs.
        configuration: Name in `model_configs.MODEL_CONFIGURATIONS` this run trained.
        description: What this run was for, in the author's own words.
        settings: Settings the run held fixed.
        curriculum: Stages it ran through.
        basis: Basis its output vector is expressed in.
        result: What the run produced.
        dataset_name: Name of the dataset it was trained on, for provenance.
        run_directory: Where the run's history and configuration were written,
            relative to the package if inside it.
        generated: Date to record; left to the caller so a rerun can pin it.

    Returns:
        The paths of the `npz` and of the `json`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    # Every stored float is float32, the precision training runs in; `load_mlp_model`
    # keeps it and `load_surrogate` casts to float64 for inference.
    arrays: dict[str, np.ndarray] = {
        f"{_PARAMS_PREFIX}{key}": np.asarray(value, dtype=np.float32)
        for key, value in flax.traverse_util.flatten_dict(
            result.params, sep="/"
        ).items()
    }
    preprocessor = result.preprocessor
    for field in preprocessing_types.PreprocessorParams.model_fields:
        value = getattr(preprocessor, field)
        if value is not None:
            arrays[f"{_PREPROCESSOR_PREFIX}{field}"] = np.asarray(
                value, dtype=np.float32
            )
    arrays.update(
        {
            f"{_BASIS_PREFIX}r_modes": np.asarray(basis.r_modes, dtype=np.int32),
            f"{_BASIS_PREFIX}z_modes": np.asarray(basis.z_modes, dtype=np.int32),
            f"{_BASIS_PREFIX}lambda_modes": np.asarray(
                basis.lambda_modes, dtype=np.int32
            ),
            f"{_BASIS_PREFIX}pressure_rho_grid": np.asarray(
                basis.pressure_rho_grid, dtype=np.float64
            ),
            f"{_BASIS_PREFIX}iota_rho_grid": np.asarray(
                basis.iota_rho_grid, dtype=np.float64
            ),
        }
    )

    metadata = {
        "kind": MLP_MODEL_KIND,
        "name": name,
        "configuration": configuration,
        "description": description,
        "generated": generated,
        "settings": settings.model_dump(),
        "curriculum": curriculum.model_dump(),
        "resolution": basis.resolution.model_dump(),
        "training": {
            "n_epochs": result.n_epochs,
            "train_mse": result.train_mse,
            "validation_total": result.validation_total,
            "train_seconds": result.train_seconds,
            "n_parameters": mlp_utils.count_parameters(result.params),
            "n_latent": preprocessor.n_latent,
            "stages": [
                {
                    "label": configured.label,
                    "n_epochs": stage.n_epochs,
                    "train_mse": stage.selected.train_data,
                    "validation_total": stage.selected.validation_total,
                    "validation_data": stage.selected.validation_data,
                    "validation_fnorm": stage.selected.validation_fnorm,
                    "selection_term": stage.selection_term,
                    "fnorm_weight": stage.fnorm_weight,
                }
                for configured, stage in zip(
                    curriculum.stages, result.stages, strict=True
                )
            ],
        },
        "provenance": {
            "dataset": dataset_name,
            "run_directory": run_directory,
            "force_normalization_constant": result.force_normalization_constant,
            "fnorm_template_equilibrium_id": result.fnorm_template_equilibrium_id,
        },
    }

    weights_path = directory / f"{name}.npz"
    metadata_path = directory / f"{name}.json"
    np.savez(weights_path, allow_pickle=False, **arrays)
    metadata_path.write_text(json.dumps(metadata, indent=4) + "\n")
    return weights_path, metadata_path


def load_surrogate(
    directory: pathlib.Path,
    name: str,
    data_directory: pathlib.Path,
    machine_scale: surrogate_types.MachineScale | None = None,
) -> LoadedSurrogate:
    """Load a stored model of either kind and hand back its forward pass.

    The two kinds have nothing in common on disk, an MLP being weights and an
    interpolator a pointer to a solved lattice, but everything that scores or plots one
    needs only a callable and the basis it predicts in. This is where they converge.

    Args:
        directory: Directory holding the stored models.
        name: Name of the model to load.
        data_directory: Directory holding the datasets, for an interpolator's lattice.
        machine_scale: Machine the forward pass is evaluated at, through
            `surrogate_utils.scaled_prediction`; None keeps the stored model's own
            normalized units.

    Returns:
        The model's forward pass, an MLP's in float64, and what a table needs to
        describe it.
    """
    metadata = load_model_metadata(directory, name)
    if metadata["kind"] == INTERPOLATOR_MODEL_KIND:
        lattice = interpolator_utils.load_lattice(
            data_directory / metadata["provenance"]["lattice_parquet"],
            data_directory / metadata["provenance"]["lattice_settings"],
        )
        return _scaled_surrogate(
            LoadedSurrogate(
                name=name,
                kind=metadata["kind"],
                basis=lattice.basis,
                predict=functools.partial(interpolator_utils.predict_y, lattice),
                n_parameters=int(np.asarray(lattice.y_lattice).size),
                metadata=metadata,
                machine_scale=surrogate_types.MachineScale(),
            ),
            machine_scale,
        )
    loaded = load_mlp_model(directory, name)
    # Inference differentiates through the network, and in float32 the objective is
    # rough at 1e-5 nats near a mode, which no line search can resolve.
    params = jax.tree.map(
        functools.partial(jnp.asarray, dtype=jnp.float64), loaded.params
    )
    preprocessor = loaded.preprocessor.model_copy(
        update={
            field: np.asarray(value, dtype=np.float64)
            for field, value in loaded.preprocessor
            if value is not None
        }
    )
    return _scaled_surrogate(
        LoadedSurrogate(
            name=name,
            kind=metadata["kind"],
            basis=loaded.basis,
            predict=functools.partial(
                mlp_utils.predict_y, loaded.mlp, params, preprocessor
            ),
            n_parameters=int(metadata["training"]["n_parameters"]),
            metadata=metadata,
            machine_scale=surrogate_types.MachineScale(),
        ),
        machine_scale,
    )


def load_mlp_model(directory: pathlib.Path, name: str) -> LoadedMLPModel:
    """Read a stored MLP surrogate back.

    Args:
        directory: Directory holding the two files.
        name: Model name, i.e. their shared stem.

    Returns:
        The network, its parameters, its normalization, its basis and its metadata.
    """
    metadata = load_model_metadata(directory, name)
    if metadata["kind"] != MLP_MODEL_KIND:
        raise ValueError(
            f"{name} is stored as a {metadata['kind']!r} model, not an MLP."
        )
    with np.load(directory / f"{name}.npz") as stored:
        arrays = {key: stored[key] for key in stored.files}

    params = flax.traverse_util.unflatten_dict(
        {
            tuple(key.removeprefix(_PARAMS_PREFIX).split("/")): jnp.asarray(
                value, dtype=jnp.float32
            )
            for key, value in arrays.items()
            if key.startswith(_PARAMS_PREFIX)
        }
    )
    preprocessor = preprocessing_types.PreprocessorParams(
        **{
            key.removeprefix(_PREPROCESSOR_PREFIX): np.asarray(value, dtype=np.float32)
            for key, value in arrays.items()
            if key.startswith(_PREPROCESSOR_PREFIX)
        }
    )
    basis = surrogate_types.EquilibriumBasisSpecification(
        resolution=desc_types.DescResolution(**metadata["resolution"]),
        r_modes=arrays[f"{_BASIS_PREFIX}r_modes"],
        z_modes=arrays[f"{_BASIS_PREFIX}z_modes"],
        lambda_modes=arrays[f"{_BASIS_PREFIX}lambda_modes"],
        pressure_rho_grid=arrays[f"{_BASIS_PREFIX}pressure_rho_grid"],
        iota_rho_grid=arrays[f"{_BASIS_PREFIX}iota_rho_grid"],
    )
    settings = mlp_types.MLPSettings(**metadata["settings"])
    return LoadedMLPModel(
        mlp=mlp_utils.build_mlp(settings.architecture, n_outputs=preprocessor.n_latent),
        params=params,
        preprocessor=preprocessor,
        basis=basis,
        metadata=metadata,
    )


def save_interpolator_model(
    directory: pathlib.Path,
    name: str,
    lattice_parquet: str,
    lattice_settings: str,
    generated: str = "",
    notes: str = "",
) -> pathlib.Path:
    """Write the entry of an equilibrium interpolator, which stores no weights.

    The interpolator is fully determined by its solved lattice, which lives in `data/`
    already, so only a `json` is written: enough for it to appear in the same registry
    as the networks it is the accuracy floor for.

    Args:
        directory: Directory to write the file into.
        name: Model name, i.e. the file stem.
        lattice_parquet: File name of the lattice parquet in `data/`.
        lattice_settings: File name of its settings JSON in `data/`.
        generated: Date to record.
        notes: Anything worth recording about this entry.

    Returns:
        The path of the `json`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / f"{name}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "kind": INTERPOLATOR_MODEL_KIND,
                "name": name,
                "generated": generated,
                "provenance": {
                    "lattice_parquet": lattice_parquet,
                    "lattice_settings": lattice_settings,
                },
                "notes": notes,
            },
            indent=4,
        )
        + "\n"
    )
    return metadata_path


def load_model_metadata(directory: pathlib.Path, name: str) -> dict:
    """Read a stored model's `json`, whichever kind of model it is.

    Args:
        directory: Directory holding the file.
        name: Model name, i.e. the file stem.

    Returns:
        The stored metadata.
    """
    return json.loads((directory / f"{name}.json").read_text())


def _scaled_surrogate(
    surrogate: LoadedSurrogate,
    machine_scale: surrogate_types.MachineScale | None,
) -> LoadedSurrogate:
    """The surrogate evaluated at `machine_scale`, or unchanged when it is None."""
    if machine_scale is None:
        return surrogate
    return surrogate._replace(
        predict=functools.partial(
            surrogate_utils.scaled_prediction,
            surrogate.predict,
            surrogate.basis,
            machine_scale,
        ),
        machine_scale=machine_scale,
    )
