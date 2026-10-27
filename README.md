# Plasma state estimation for diagnostic placement optimization

Code of the Master's thesis *Bayesian Experimental Design of Fusion Plasma Diagnostics
with Self-Consistent MHD Equilibria* (Technical University of Munich, 2026).

## Abstract

Whether a fusion plasma reaches ignition, in which the fusion reactions sustain its
temperature without external heating, is commonly assessed by the triple product of its
density, temperature and energy confinement time. A precise calculation of the triple
product takes into account the profiles of the electron density ($n_e$) and temperature
($T_e$) across the plasma, referred to in this work as the plasma state. These profiles
together set the plasma pressure, which in turn determines the magnetohydrodynamic (MHD)
equilibrium, the nested magnetic flux surfaces that confine the plasma. Since plasma
diagnostics cannot measure the equilibrium or the pressure directly, obtaining the
plasma state from the diagnostics signals is an inverse problem. This is commonly solved
by Bayesian inference, which yields the plasma state together with its uncertainty.

How precisely the plasma state is inferred depends on the design and placement of the
diagnostics, which Bayesian experimental design ranks by their expected information gain
of their measurements on the plasma state. Building on work at Wendelstein 7-X (W7-X),
the world's largest experimental stellarator, this work develops a general diagnostic
design method that extends previous approaches in three main ways. First, the placement
of several diagnostics is optimized jointly rather than one diagnostic at a time.
Second, a physics-informed neural-network surrogate approximates the MHD equilibrium of
each plasma state from its pressure in milliseconds. The design thereby accounts for the
displacement of the flux surfaces and for the different plasma states expected in
operation. Third, the readings predicted by the diagnostic models are linearized about
each expected plasma state, that is, approximated through their sensitivity to small
changes of the profiles. The expected information gain then follows in closed form from
these sensitivities, which makes the evaluation of each candidate layout faster than
with the Monte Carlo sampling of previous approaches.

The method is applied to a Thomson scattering system, which measures $n_e$ and $T_e$
locally, and an interferometer, which measures the line-integrated $n_e$. The plasma
configuration is quasi-isodynamic and shares the main properties of Alpha, the
stellarator demonstrator of Proxima Fusion and the Max Planck Institute for Plasma
Physics. Relative to the initial layout, the joint placement lowers the errors of the
inferred $n_e$ and $T_e$ profiles by 34.9 and 39.0% and narrows their uncertainty bands
by 53.0 and 13.9%. Inferring each plasma state with its own equilibrium, instead of a
fixed one, lowers these errors by 16.6 and 22.2%. Finally, one calibration correction
per polychromator is inferred with the profiles, which could reveal a drift of the
Thomson scattering calibration and reduce the recalibration downtime of large fusion
experiments. This work thereby provides a fast and integrated method to design and place
plasma diagnostics within the stellarator design cycle.

## Structure

The package lives in `src/plasma_state_estimation_for_diagnostic_placement_optimization/`;
the datasets in the top-level `data/`.

| Folder | Content |
|---|---|
| `ideal_mhd/` | DESC Fourier-Zernike geometry evaluation and the xyz to flux-coordinate Newton solve |
| `inverse_mapping/` | Differentiable inverse map (implicit-function-theorem derivatives) and its test equilibria |
| `mhd_equilibrium_forward_models/` | Equilibrium surrogates (MLP, trilinear interpolator): datasets, training, hyperparameter search, trained models |
| `diagnostics_forward_models/` | Thomson scattering and interferometer synthetic diagnostics, chord geometry, the pinned design point |
| `bayesian_inference/` | Profile parametrizations, priors, likelihood, NumPyro model, MAP, Laplace approximation and NUTS |
| `diagnostics_placement_optimization/` | Expected information gain of a diagnostic layout |
| `results_plots_tables/` | One folder per study (run script, notebook, figures, tables) and the shared plotting and sweep utilities |
| `data/` | Datasets downloaded from Hugging Face and a notebook to explore them, see `data/README.md` |

## Data and trained models

The datasets and the trained surrogates are on Hugging Face, under CC BY 4.0, and the
installation below downloads them at the revision the thesis used:

| Repository | Content |
|---|---|
| [diagnostics-bayesian-experimental-design-datasets](https://huggingface.co/datasets/proxima-fusion/diagnostics-bayesian-experimental-design-datasets) | The 3000 DESC equilibria the surrogates are trained on and the 28-node lattice of the equilibrium interpolator, each a parquet with a settings JSON (basis, mode tables, radial grids, train/val/test split), and the DESC starting point every row was solved from |
| [diagnostics-bayesian-experimental-design-surrogates](https://huggingface.co/proxima-fusion/diagnostics-bayesian-experimental-design-surrogates) | Weights of the two MLP surrogates and their training histories |

`data/README.md` describes how the equilibria were generated and how to regenerate one.

## Installation

Needs Python 3.12, the version the pinned packages were tested with.

```
gh repo clone matildevalente-pf/diagnostics_bayesian_experimental_design
cd diagnostics_bayesian_experimental_design
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[test,notebooks]"
python -m ipykernel install --user --name diagnostics-bed --display-name "diagnostics-bed (Python 3.12)"
hf auth login        # while the Hugging Face repositories are private
python src/plasma_state_estimation_for_diagnostic_placement_optimization/mhd_equilibrium_forward_models/download_from_hugging_face.py
```

`requirements.txt` pins the direct dependencies at the versions the thesis results were
computed with, and `-e` installs the package in editable mode, so changes under `src/`
apply without reinstalling. The kernel `diagnostics-bed` runs the notebooks. The download
keeps files that already exist. If pip has to build `netcdf4` from source because no
wheel exists for your platform, install the netCDF C library first
(`sudo apt-get install -y libnetcdf-dev` or `brew install netcdf`).

## Reproducing the thesis results

Every command runs from this folder with the environment active. Notebook paths are
relative to
`src/plasma_state_estimation_for_diagnostic_placement_optimization/`, except those under
`data/`. Open each notebook with the `diagnostics-bed` kernel and run all cells.

```
S=src/plasma_state_estimation_for_diagnostic_placement_optimization
R=$S/results_plots_tables/results
E=$R/plasma_state_estimation
P=$R/diagnostics_placement_optimization
```

### 1. Dataset

Notebook: `data/explore_dataset.ipynb`

```
# One row solved again with DESC: about 7 min, 10 GB;
# expect max differences ~1.5e-14, 2.2e-14, 1.7e-13
python $S/mhd_equilibrium_forward_models/regenerate_dataset_row.py --row-id DgSADUTGs7s2ZBxE5xge2ej
```

### 2. Methods chapter

Notebook only: `results_plots_tables/results/diagnostics_forward_models/diagnostics_forward_models.ipynb`

```
python $R/inverse_mapping/stress_tests/run_soak.py
```

Notebook: `results_plots_tables/results/inverse_mapping/inverse_mapping.ipynb`

```
python $R/mhd_equilibrium_forward_models/test_set_evaluation/run_test_set_evaluation.py \
    --models equilibrium_interpolator 20260908_141134_mlp_data_pca_hand_tuned \
    20260907_081512_mlp_data_fnorm_only_pca_hand_tuned --n-workers 2
```

Notebook: `results_plots_tables/results/mhd_equilibrium_forward_models/test_set_evaluation/results.ipynb`

```
# 480 DESC solves, several hours (can be left for last)
python $R/mhd_equilibrium_forward_models/desc_resolution_analysis/run_desc_resolution_analysis.py --n-workers 2
```

Notebook: `results_plots_tables/results/mhd_equilibrium_forward_models/desc_resolution_analysis/desc_resolution_analysis.ipynb`

### 3. Plasma state estimation (Sec. 3.1)

After each study, run the `results.ipynb` in `results_plots_tables/results/plasma_state_estimation/<study>/`.

```
# 3.1.1 mcmc_benchmark (the NUTS arm takes 2 to 3 h per run; good overnight)
python $E/mcmc_benchmark/run_study.py --arms map_laplace --workers 4
python $E/mcmc_benchmark/run_study.py --arms mcmc_nuts --workers 4

# 3.1.2 prior_likelihood_posterior
python $E/prior_likelihood_posterior/run_study.py --workers 4

# 3.1.3 ts_band_scale
python $E/ts_band_scale/run_study.py --workers 4

# 3.1.4 self_consistency_analysis
python $E/self_consistency_analysis/run_study.py --n-seeds 5 --workers 4

# 3.1.5 ts_uncertainty_level_vs_inference (336 runs, a few hours)
python $E/ts_uncertainty_level_vs_inference/run_study.py --workers 4
```

### 4. Diagnostic placement (Sec. 3.2, strictly in this order)

```
# 3.2.1
python $P/if_placement_study/run_study.py
python $P/if_placement_study/run_study.py --chord-counts 1 2 3 4 5 6 \
    --output $P/if_placement_study/tables/if_placement_scan_chord_counts.csv     # about 3 h
python $P/run_placement_inference.py --study if_placement_study --workers 4
python $P/run_placement_inference.py --study if_placement_study \
    --scan $P/if_placement_study/tables/if_placement_scan_chord_counts.csv \
    --layouts 3_chords_optimized 4_chords_optimized 5_chords_optimized 6_chords_optimized --workers 4
```

Notebook: `results_plots_tables/results/diagnostics_placement_optimization/if_placement_study/results.ipynb`

```
# 3.2.2 to 3.2.4: all four scans before any inference
python $P/if_and_ts_placement_study/run_scan.py --laser-through-axis                         # 1390 layouts, best 37.43 bit
python $P/if_and_ts_placement_study/run_scan.py --laser-through-axis --thomson-only          # best 32.44 bit
python $P/if_and_ts_placement_study/run_scan.py --laser-through-axis --interferometer-only   # 15 rows, best 14.03 bit
python $P/if_and_ts_placement_study/run_scan.py --laser-through-axis --fixed-equilibrium     # same optimum as the first scan
python $P/run_placement_inference.py --study if_and_ts_placement_study --workers 4
python $P/run_placement_inference.py --study if_and_ts_placement_study \
    --layouts initial_guess optimized separately_optimized sequentially_optimized \
    --n-seeds 5 --skip-stored --workers 4
python $P/run_placement_inference.py --study if_and_ts_placement_study \
    --layouts initial_guess optimized \
    --truths test_state_1_parametric test_state_2_parametric test_state_3_parametric \
    test_state_4_parametric test_state_5_gp test_state_6_gp test_state_7_gp test_state_8_gp \
    --n-seeds 5 --workers 4
```

Notebook: `results_plots_tables/results/diagnostics_placement_optimization/if_and_ts_placement_study/results.ipynb`

One worker needs about 2.5 GB of memory. `--skip-stored` resumes an interrupted study
without repeating its finished runs (all but `self_consistency_analysis`).

### 5. Optional: the surrogates themselves

The download provides the trained surrogates, so the studies above do not need this
step. It reproduces the hyperparameter search, which writes the run folder that
`results_plots_tables/results/mhd_equilibrium_forward_models/hyperparameter_optimization/results.ipynb`
reads, and the training of the two surrogates.

```
H=$S/mhd_equilibrium_forward_models/hyperparameter_optimization
python $H/hyperparameter_optimization_utils.py                 # grid search, five trainings at a time
python $H/hyperparameter_optimization_utils.py --force-stage   # both stages for the selected nodes
python $H/pca_component_sweep.py --n-workers 2
python $S/mhd_equilibrium_forward_models/train_surrogate.py \
    --model mlp_data_pca_hand_tuned --description "MLP"            # about 6 min
python $S/mhd_equilibrium_forward_models/train_surrogate.py \
    --model mlp_data_fnorm_only_pca_hand_tuned --description "PINN"   # about 20.5 h
```

A retrained surrogate is stored under a new timestamped name; pass it to a study with
`--model-name` to use it instead of the downloaded one.

## Tests

```
pytest .                    # 99 unit tests, about 2 min, no data needed
pytest . -k <test_name>     # one test
pytest . -m integration     # 2 slow tests, need the download of the installation
```
