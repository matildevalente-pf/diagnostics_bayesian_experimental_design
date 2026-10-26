# Datasets

Datasets of fixed-boundary DESC equilibria for a quasi-isodynamic stellarator
configuration (NFP = 3, stellarator symmetric) with the corresponding pressure profiles.
The configuration is scaled to B0 = 1 T. The Fourier-Zernike resolution used was
L = M = 11, N = 12. The parquets, their `_settings.json` and
`equilibrium_a_warm_start.json` are downloaded from Hugging Face (see the top-level
`README.md`). A settings JSON holds the basis and, for the 3k dataset, the
train/val/test split, which `equilibrium_a_3k_spline_input_split.json` repeats.

| Dataset | Rows | Pressure profiles |
|---|---|---|
| `equilibrium_a_3k_spline_input` | 3000 | 2400 Gaussian-process and 600 parametric `p0 (1 - s^gamma)^alpha`, `p0 <= 35` kPa. Trains the MLP surrogates. |
| `equilibrium_a_interpolator_28_spline_input` | 28 | Lattice of `p0 (1 - rho^a)^b`: `p0` in {0, 11.7, 23.3, 35} kPa, `a` in {2, 4, 6}, `b` in {0.5, 2.25, 4}. Builds the equilibrium interpolator. |

Columns: `pressure_knots` and `pressure_params` (the pressure spline the solver saw, Pa;
`pressure` repeats the values), `Psi` (Wb), `R_lmn`, `Z_lmn`, `lambda_lmn`, `iotaf`,
`beta_volume_averaged`, the `fnorm_volume_averaged_*` force residual, and
`desc_equilibrium_id` (row id).

## Generation

A vacuum equilibrium was computed once with VMEC (fixed boundary), converted to DESC at
L = M = 11, N = 12 and solved again with DESC. Every row is then one DESC
`Equilibrium.solve(objective="force")`, `lsq-exact` (trust-region least squares) at
DESC's default tolerances, started from that vacuum equilibrium with the row's pressure
profile. `equilibrium_a_warm_start.json` holds this starting point.

| Dataset | Wall time | Machine |
|---|---|---|
| `equilibrium_a_3k` | 2 h 23 min | 250 cloud workers in parallel |
| `equilibrium_a_interpolator_28` | 1 h 25 min (252 to 2122 s per solve) | AMD EPYC 7B12, 32 vCPU, 125 GB RAM, 7 solves in parallel |

## Regenerating a row

```
cd src/plasma_state_estimation_for_diagnostic_placement_optimization
python mhd_equilibrium_forward_models/regenerate_dataset_row.py --row-id <desc_equilibrium_id>
```

solves one row again from the starting point (about 7 min and 10 GB of memory) and
prints its largest difference from the stored `R_lmn`, `Z_lmn` and `lambda_lmn`.
`iotaf`, `beta_volume_averaged` and the `fnorm_*` columns are evaluated on the solved
equilibrium.
