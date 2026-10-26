# The MLP surrogates the results compare, as named configurations: the data fit, and
# the same network trained further on the force residual alone.

from typing import NamedTuple

from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    mlp_types,
)

N_PCA_COMPONENTS = 1244
"""PCA components the target keeps, where a model uses PCA.

A fixed count, never an explained-variance fraction, and the width every surrogate in
`trained_models/` was trained at.
"""

HAND_TUNED_LEARNING_RATE = 1e-3
"""First-stage learning rate of the 2026-08-29 force models; their force stage runs a
tenth of it."""

HAND_TUNED_SETTINGS = mlp_types.MLPSettings(
    architecture=mlp_types.MLPArchitecture(
        hidden_layer_widths=[512] * 3, activation="leaky_relu", dropout_rate=0.0
    ),
    n_pca_components=N_PCA_COMPONENTS,
    weight_decay=0.0,
    batch_size=32,
)
"""The hand-picked settings of `20260829_020204_mlp_data_fnorm_only_pca`, which train the
force stage far better than the searched ones, so both configurations use them."""

DATA_STAGE = mlp_types.CurriculumStepConfig(
    loss_config=mlp_types.FinetuneLossConfig(use_data=True),
    n_epochs=2000,
    patience=200,
    min_epochs=500,
    learning_rate=HAND_TUNED_LEARNING_RATE,
    label="s1_data",
)
"""The first stage of both models: the plain data fit, and the whole of
`mlp_data_pca_hand_tuned`."""

FNORM_ONLY_STAGE = mlp_types.CurriculumStepConfig(
    # The 2026-08-29 run weighted the lone force term by one, so its validation term
    # reads as the logarithm of the residual itself.
    loss_config=mlp_types.FinetuneLossConfig(
        use_data=False, use_fnorm=True, loss_fnorm_weight=1.0
    ),
    n_epochs=2000,
    patience=100,
    min_epochs=200,
    learning_rate=HAND_TUNED_LEARNING_RATE / 10.0,
    label="s2_fnorm_only",
    selection_term="fnorm",
)
"""The force residual alone, with the data term switched off rather than outweighed, at a
tenth of the first stage's learning rate."""


class ModelConfiguration(NamedTuple):
    """One named model: what it holds fixed, and the stages it is trained through."""

    settings: mlp_types.MLPSettings
    """Architecture, optimizer and target-space settings."""

    curriculum: mlp_types.CurriculumLearningConfig
    """The stages, in order."""

    description: str
    """One line on what this model is in the comparison for."""


MODEL_CONFIGURATIONS: dict[str, ModelConfiguration] = {
    "mlp_data_pca_hand_tuned": ModelConfiguration(
        settings=HAND_TUNED_SETTINGS,
        curriculum=mlp_types.CurriculumLearningConfig(stages=[DATA_STAGE]),
        description=(
            "The data stage of the hand-tuned settings alone: the force-trained "
            "hand-tuned models before their force stage"
        ),
    ),
    "mlp_data_fnorm_only_pca_hand_tuned": ModelConfiguration(
        settings=HAND_TUNED_SETTINGS,
        curriculum=mlp_types.CurriculumLearningConfig(
            stages=[DATA_STAGE, FNORM_ONLY_STAGE]
        ),
        description=(
            "The 2026-08-29 recipe, force residual alone, on the stored pressure spline"
        ),
    ),
}
"""The models the results compare, keyed by the name they are stored under."""
