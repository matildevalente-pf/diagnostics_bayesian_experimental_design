# Datasets

Fixed-boundary DESC equilibria on the boundary of equilibrium A, scaled to B0 = 1 T
(NFP = 3, stellarator symmetric, L = M = 11, N = 12), one row per pressure profile. The
parquets and their `_settings.json` are downloaded from Hugging Face (see the top-level
`README.md`). A settings JSON holds the basis and, for the 3k dataset, the
train/val/test split, which `equilibrium_a_3k_spline_input_split.json` repeats.

| Dataset | Rows | Pressure profiles |
|---|---|---|
| `equilibrium_a_3k_spline_input` | 3000 | 2400 Gaussian-process and 600 parametric `p0 (1 - s^gamma)^alpha`, `p0 <= 35` kPa. Trains the MLP surrogates. |
| `equilibrium_a_interpolator_28_spline_input` | 28 | Lattice of `p0 (1 - rho^a)^b`: `p0` in {0, 11.7, 23.3, 35} kPa, `a` in {2, 4, 6}, `b` in {0.5, 2.25, 4}. Builds the equilibrium interpolator. |

Columns: `pressure_knots` and `pressure_params` (the pressure spline the solver saw, Pa;
`pressure` repeats the values), `Psi` (Wb), `R_lmn`, `Z_lmn`, `lambda_lmn`, `iotaf`,
`beta_volume_averaged`, the `fnorm_volume_averaged_*` force residual, and
`desc_equilibrium_id` (provenance).

## Generation

Each row is one DESC `Equilibrium.solve(objective="force")`, `lsq-exact` (trust-region
least squares) at DESC's default tolerances, warm started from the shared vacuum
equilibrium.

| Dataset | Wall time | Machine |
|---|---|---|
| `equilibrium_a_3k` | 2 h 23 min | 250 cloud workers in parallel |
| `equilibrium_a_interpolator_28` | 1 h 25 min (252 to 2122 s per solve) | AMD EPYC 7B12, 32 vCPU, 125 GB RAM, 7 solves in parallel |

## Regenerating a row

```python
import desc.profiles

eq = vacuum.copy()  # vacuum warm start, provenance id D2LKaPDSGCuGTjv2Zcj9g8A
eq.surface = boundary  # provenance id DRnvsp6pzgmPZWwXdF6GDCG, check_orientation=True
eq.pressure = desc.profiles.SplineProfile(knots=row.pressure_knots, values=row.pressure_params)
eq.current = desc.profiles.SplineProfile(knots=row.pressure_knots, values=0 * row.pressure_params)
eq.iota = None
eq.Psi = row.Psi
eq.solve(objective="force")
```

`iotaf`, `beta_volume_averaged` and the `fnorm_*` columns are then evaluated on the
solved `eq`.
