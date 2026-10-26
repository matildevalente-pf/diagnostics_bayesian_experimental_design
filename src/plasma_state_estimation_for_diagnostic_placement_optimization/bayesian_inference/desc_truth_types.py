# The ground-truth states the synthetic data is generated from: real DESC solves
# read from the training parquet, with their pressure split into `ne` and `Te`.

import pathlib

import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import pyarrow.compute
import pyarrow.parquet
import pydantic
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    diagnostic_design_point,
    thomson_scattering_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_utils,
    surrogate_types,
)

DESC_TRUTH_IDS: dict[str, str] = {
    "peaked_1_parametric": "DmLGKGELiHcSMXuZNXpMsFb",
    "peaked_2_parametric": "DmNkmmbkA8zfRnpm8vkwjXn",
    "wide_1_parametric": "DBSkSZ7mo9bbPVP4GLkK9ga",
    "wide_2_parametric": "DnsQqKCcECYexCDs9TQeM3g",
    "wide_low_beta_gp": "DMC34UGZA9r9GTXbbsY8ZFt",
    "wide_high_beta_gp": "D2potSHWmkq42DSX8pyBdqW",
    "slightly_hollow_gp": "DdKJ3mMN7oERbftCT32wPgg",
    "hollow_gp": "DFSMW6RHiNMKruq2WeGPHXL",
}
"""Named ground truths, by the `desc_equilibrium_id` their `equilibrium_a_3k` row
carries, all in the pinned test split at beta 1.5 to 3.6 % and core Te up to 10 keV;
the suffix is the sampling family, read off the two-power fit residual."""

TEST_STATE_IDS: dict[str, str] = {
    "test_state_1_parametric": "D9FaxPYoLKVppr8aUSnAegE",
    "test_state_2_parametric": "Dn4vML8fu8i7iEY4kd38BwR",
    "test_state_3_parametric": "DPQuawYDdR3yooj3rYuGT5K",
    "test_state_4_parametric": "DnbsLj3ZfYEAATZRucdhnQY",
    "test_state_5_gp": "DhU4CppENhXDcGaZ9CB3c62",
    "test_state_6_gp": "DQX8JAPwUJaAfpcG4G7GgmH",
    "test_state_7_gp": "Dg4H3DzsCMCyLpN5guhQJ7W",
    "test_state_8_gp": "DVgKXgdTcQrZoV8XEe2Bc6c",
}
"""Test-split states no layout was optimized on, to check that an optimized layout
also serves them: four of each sampling family, drawn with
`numpy.random.default_rng(0)` from the 82 test rows outside `DESC_TRUTH_IDS` inside the
windows that catalogue was chosen from (peak `ne` 1e20 to 9e20 m^-3, peak `Te` 1 to
10 keV, beta 1.5 to 4 %, non-negative pressure) whose two-power fit lies inside the
prior, 10 of them parametric; the suffix is the family, as in `DESC_TRUTH_IDS`."""


REFERENCE_EQUILIBRIUM_ID = "DdgGq5HxCvF9qgAM7HDLCKZ"
"""The equilibrium a reconstruction may assume before the profiles are known.

The median of the **training** split by `beta_volume_averaged` among the rows whose
pressure is non-negative everywhere (2.06%, against the truths' 1.52% to 3.61%), so it
is plausible and wrong for every truth in both directions. The self-consistency study's
fixed arm assumes it, and the Thomson band-scale rule reads its expected temperature.
"""

_MINIMUM_PRESSURE_PA = 1.0
"""Floor applied before taking logs: dataset pressure profiles can graze zero."""


class DescTruth(pydantic.BaseModel):
    """One ground-truth state: a DESC solve's geometry plus the `ne`/`Te` split from it.

    A solve constrains only the total pressure, so the split is a modelling assumption;
    see `profiles_from_pressure_spline`. `DESC_TRUTH_IDS[name]` recovers the source id.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: str
    """Catalog name, a key of `DESC_TRUTH_IDS`, or a free label for an ad hoc truth."""

    coefficients: desc_types.DescFourierZernikeCoefficients
    """Geometry the synthetic diagnostics are placed on and inverted against."""

    profiles: profile_parametrization_types.PlasmaProfiles
    """`spline_knots` `ne`/`Te` reproducing the solved pressure at the spline's own
    knots, which are the surrogate's input grid."""

    machine_scale: surrogate_types.MachineScale = surrogate_types.MachineScale()
    """The machine the geometry and the pressure were carried to; unit factors for the
    dataset's own `R0 = 1 m`, `B = 1 T`."""


def read_desc_truth(
    name_or_id: str,
    parquet_path: pathlib.Path,
    settings_path: pathlib.Path,
    name: str = "",
    machine_scale: surrogate_types.MachineScale | None = (
        diagnostic_design_point.MACHINE_SCALE
    ),
) -> DescTruth:
    """Read one ground truth from the training parquet, by catalog name or by id, at the
    machine scale unless told otherwise.

    A key of `DESC_TRUTH_IDS` or `TEST_STATE_IDS` resolves to its id; anything else is
    taken to be an id already. Only the matching row and the columns a truth needs are
    read.

    Args:
        name_or_id: A key of `DESC_TRUTH_IDS` or `TEST_STATE_IDS`, or a
            `desc_equilibrium_id`.
        parquet_path: Path of the `equilibrium_a_3k` parquet.
        settings_path: Path of its settings JSON, which carries the mode tables.
        name: Label to record on the truth; `name_or_id` when left empty.
        machine_scale: Factors the stored `R`, `Z` (times `L`) and pressure (times
            `b^2`) are carried to the machine with; None reads the row as stored.

    Returns:
        The truth, ready to place diagnostics on.
    """
    scale = machine_scale or surrogate_types.MachineScale()
    equilibrium_id = (DESC_TRUTH_IDS | TEST_STATE_IDS).get(name_or_id, name_or_id)
    basis = dataset_utils.load_metadata(settings_path).basis
    table = pyarrow.parquet.read_table(
        parquet_path,
        columns=[
            "desc_equilibrium_id",
            "R_lmn",
            "Z_lmn",
            "lambda_lmn",
            "pressure_knots",
            "pressure_params",
        ],
        filters=pyarrow.compute.field("desc_equilibrium_id") == equilibrium_id,
    )
    if table.num_rows != 1:
        raise ValueError(
            f"{parquet_path.name} holds {table.num_rows} rows with "
            f"desc_equilibrium_id {equilibrium_id} (from {name_or_id!r}); expected 1. "
            f"The catalog names are {sorted(DESC_TRUTH_IDS)}."
        )
    row = table.to_pylist()[0]
    return DescTruth(
        name=name or name_or_id,
        coefficients=desc_types.DescFourierZernikeCoefficients(
            r_lmn=np.asarray(row["R_lmn"], dtype=np.float64) * scale.length_factor,
            z_lmn=np.asarray(row["Z_lmn"], dtype=np.float64) * scale.length_factor,
            lambda_lmn=np.asarray(row["lambda_lmn"], dtype=np.float64),
            r_modes=np.asarray(basis.r_modes),
            z_modes=np.asarray(basis.z_modes),
            lambda_modes=np.asarray(basis.lambda_modes),
            n_field_periods=basis.resolution.n_field_periods,
        ),
        profiles=profiles_from_pressure_spline(
            np.asarray(row["pressure_knots"], dtype=np.float64),
            np.asarray(row["pressure_params"], dtype=np.float64)
            * scale.pressure_factor,
        ),
        machine_scale=scale,
    )


def profiles_from_pressure_spline(
    pressure_knots: jt.Float[NpOrJaxArray, " n_pressure_knot"],
    pressure_params: jt.Float[NpOrJaxArray, " n_pressure_knot"],
) -> profile_parametrization_types.PlasmaProfiles:
    """The `ne`/`Te` pair a solved pressure spline implies, as `spline_knots`.

    A solve constrains only `p = 2 e ne Te`, so the split is a modelling assumption:
    `Te = TE_REFERENCE_EV sqrt(p / PRESSURE_REFERENCE_PA)`, `ne = p / (2 e Te)`.
    The profiles live on the pressure spline's own knots, which are also the surrogate's
    input grid, so the pressure the surrogate is handed for a truth is the solved one.

    Args:
        pressure_knots: Radial knots of the solved pressure spline, in `rho`.
        pressure_params: Pressure at those knots, in Pa.

    Returns:
        The `spline_knots` `ne`/`Te` pair.
    """
    # The `(1 - rho^2)` envelope is zero on the boundary, so the knot there would need
    # the log of zero; the envelope carries both profiles to zero at `rho = 1` anyway.
    inside = jnp.asarray(pressure_knots) < 1.0
    rho_knots = jnp.asarray(pressure_knots)[inside]
    pressure_at_knots = jnp.maximum(
        jnp.asarray(pressure_params)[inside], _MINIMUM_PRESSURE_PA
    )
    temperature_at_knots = diagnostic_design_point.TE_REFERENCE_EV * jnp.sqrt(
        pressure_at_knots / diagnostic_design_point.PRESSURE_REFERENCE_PA
    )
    density_at_knots = pressure_at_knots / (
        2.0 * thomson_scattering_utils.ELEMENTARY_CHARGE * temperature_at_knots
    )
    # The envelope is divided out at the knots, so the evaluated profile reproduces the
    # target values there exactly rather than damped by `(1 - rho^2)` twice.
    envelope = 1.0 - jnp.square(rho_knots)
    return profile_parametrization_types.PlasmaProfiles(
        family="spline_knots",
        electron_density_params=jnp.log(density_at_knots / envelope),
        electron_temperature_params=jnp.log(temperature_at_knots / envelope),
        rho_knots=rho_knots,
    )
