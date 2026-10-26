# Plasma state estimation for diagnostic placement optimization

Code of the Master's thesis *Bayesian Experimental Design of Fusion Plasma Diagnostics
with Self-Consistent MHD Equilibria* (Technical University of Munich, 2026).

## Abstract

The confinement of a fusion plasma is characterized by its stored energy and energy
confinement time. Both are derived from the plasma state, characterized by its pressure
profile, and from the magnetohydrodynamic (MHD) equilibrium that this pressure modifies.
Since plasma diagnostics measure neither directly, the plasma state and its equilibrium
are inferred from the diagnostic readings. This inverse problem is commonly solved by
Bayesian inference, which yields the plasma state together with its uncertainty.

How precisely the plasma state is inferred depends on the design and placement of the
diagnostics. Bayesian experimental design ranks candidate designs by their expected
information gain, the information their measurements are expected to provide on the
plasma state. At W7-X, it was used to optimize individual diagnostics, with the gain
estimated by Monte Carlo sampling. This work develops a general diagnostic design method
that extends this and other previous approaches in three main ways. First, the placement
of several diagnostics is optimized jointly rather than one diagnostic at a time. Second,
a physics-informed neural-network surrogate approximates the MHD equilibrium of each
plasma state from its pressure in milliseconds. The design thereby accounts for the
displacement of the flux surfaces and for the different plasma states expected in
operation. Third, every model is JAX traceable, so that the placement optimization and
the inference may use automatic differentiation.

This method is applied to Alpha, the quasi-isodynamic stellarator demonstrator under
development by Proxima Fusion with the Max Planck Institute for Plasma Physics. The
plasma configuration used is not that of Alpha itself, but a quasi-isodynamic
configuration that shares its main properties. The diagnostics designed and optimized
are a Thomson scattering system, which measures the local electron density ($n_e$) and
temperature ($T_e$), and an interferometer, which provides line integrated $n_e$. The
results show that the joint placement optimization of both diagnostics, relative to the
initial layout, lowers the errors of the inferred $n_e$ and $T_e$ profiles by 13.9 and
20.7% and narrows their uncertainty bands by 57.8 and 15.2%. Additionally, inferring
each plasma state with its own equilibrium from the surrogate, instead of a fixed
equilibrium, lowers the $n_e$ and $T_e$ errors by 16.6 and 22.2%, since the displacement
of the flux surfaces by the plasma pressure is then accounted for. Finally, the coating
and degradation of optical components and drifts of the detector gain change the
calibration of a Thomson scattering system over time. One correction factor per
polychromator is therefore inferred together with the profiles. In large fusion
experiments, where a dedicated calibration requires access to the torus hall, these
corrections could reveal a calibration drift and thereby reduce the downtime spent on
recalibration. This work thereby provides a fast and integrated method to design and
place plasma diagnostics within the stellarator design cycle.

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

The datasets and the trained surrogates are on Hugging Face, under CC BY 4.0:

| Repository | Content |
|---|---|
| [diagnostics-bayesian-experimental-design-datasets](https://huggingface.co/datasets/proxima-fusion/diagnostics-bayesian-experimental-design-datasets) | The 3000 DESC equilibria the surrogates are trained on and the 28-node lattice of the equilibrium interpolator, each a parquet with a settings JSON (basis, mode tables, radial grids, train/val/test split), and the DESC starting point every row was solved from |
| [diagnostics-bayesian-experimental-design-surrogates](https://huggingface.co/proxima-fusion/diagnostics-bayesian-experimental-design-surrogates) | Weights of the two MLP surrogates and their training histories |

After the installation below, download them into place at the revision the thesis used.
Files that already exist are kept:

```
python src/plasma_state_estimation_for_diagnostic_placement_optimization/mhd_equilibrium_forward_models/download_from_hugging_face.py
```

While the repositories are private, log in first with `hf auth login`.
`data/explore_dataset.ipynb` loads the training dataset, and `data/README.md` describes
how the equilibria were generated and how to regenerate one.

## Installation

Needs Python 3.12 or later and the netCDF C library, which DESC uses through `netcdf4`:

```
sudo apt-get install -y libnetcdf-dev   # Ubuntu / Debian
brew install netcdf                     # macOS
```

Then, from this folder:

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[test,notebooks]"
```

`requirements.txt` pins the versions the thesis results were computed with. `-e`
installs the package in editable mode, so changes under `src/` apply without
reinstalling. The `test` extra adds pytest and `notebooks` adds Jupyter for the
notebooks under `results_plots_tables/` and `data/`.

To skip the system library, use the Docker image instead. It holds the pinned versions,
and the checkout is mounted at `/workspace`, so downloads and results land in it and a
change to `src/` needs no rebuild:

```
docker build -t diagnostics-bayesian-experimental-design .
docker run --rm -it --memory 24g --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$HOME/.cache/huggingface":/tmp/.cache/huggingface -v "$(pwd)":/workspace \
    diagnostics-bayesian-experimental-design python path/to/script.py
```

`--memory` stops a container that outgrows it instead of the host, and the Hugging Face
mount passes in the login of `hf auth login`.

## Tests

From this folder:

```
pytest .                    # unit tests, about 2 min, no data needed
pytest . -k <test_name>     # one test
pytest . -m integration     # slow tests
```

The integration tests need the dataset and the surrogates downloaded first (see
"Data and trained models").

## Reproducing the thesis results

Notebook paths are relative to `src/plasma_state_estimation_for_diagnostic_placement_optimization/`,
except `data/explore_dataset.ipynb`. Open each in VS Code or Jupyter with the
`diagnostics-bed` kernel and run all cells.

### 1. Clone and install

```
gh repo clone matildevalente-pf/diagnostics_bayesian_experimental_design
cd diagnostics_bayesian_experimental_design

sudo apt-get install -y libnetcdf-dev      # Ubuntu; on macOS: brew install netcdf
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[test,notebooks]"
python -m ipykernel install --user --name diagnostics-bed --display-name "diagnostics-bed (Python 3.12)"

# must print a path inside this folder
python -c "import plasma_state_estimation_for_diagnostic_placement_optimization as p; print(p.__path__)"
```

### 2. Shortcuts for this terminal

```
S=src/plasma_state_estimation_for_diagnostic_placement_optimization
R=$S/results_plots_tables/results
E=$R/plasma_state_estimation
P=$R/diagnostics_placement_optimization
```

### 3. Tests, data and one regenerated row

```
pytest .                                   # 106 passed
hf auth login
python $S/mhd_equilibrium_forward_models/download_from_hugging_face.py   # 13 files
pytest . -m integration                    # 3 passed

# about 7 min, 10 GB; expect max differences ~1.5e-14, 2.2e-14, 1.7e-13
python $S/mhd_equilibrium_forward_models/regenerate_dataset_row.py --row-id DgSADUTGs7s2ZBxE5xge2ej
```

Notebook: `data/explore_dataset.ipynb`

### 4. Methods chapter

Notebook only: `results_plots_tables/results/diagnostics_forward_models/diagnostics_forward_models.ipynb`

```
python $R/inverse_mapping/stress_tests/run_soak.py
```

Notebook: `results_plots_tables/results/inverse_mapping/inverse_mapping.ipynb`

```
python $R/mhd_equilibrium_forward_models/test_set_evaluation/run_test_set_evaluation.py \
    --models equilibrium_interpolator 20260908_141134_mlp_data_pca_hand_tuned \
    20260907_081512_mlp_data_fnorm_only_pca_hand_tuned --n-workers 4
```

Notebook: `results_plots_tables/results/mhd_equilibrium_forward_models/test_set_evaluation/results.ipynb`

```
# 480 DESC solves, several hours (can be left for last)
python $R/mhd_equilibrium_forward_models/desc_resolution_analysis/run_desc_resolution_analysis.py --n-workers 2
```

Notebook: `results_plots_tables/results/mhd_equilibrium_forward_models/desc_resolution_analysis/desc_resolution_analysis.ipynb`

### 5. Plasma state estimation (Sec. 3.1)

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
python $E/self_consistency_analysis/run_study.py --workers 4

# 3.1.5 ts_uncertainty_level_vs_inference (336 runs, a few hours)
python $E/ts_uncertainty_level_vs_inference/run_study.py --workers 4
```

### 6. Diagnostic placement (Sec. 3.2, strictly in this order)

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
# 3.2.2 to 3.2.4: all three scans before any inference
python $P/if_and_ts_placement_study/run_scan.py --laser-through-axis                         # 1390 layouts, best 37.43 bit
python $P/if_and_ts_placement_study/run_scan.py --laser-through-axis --thomson-only          # best 32.44 bit
python $P/if_and_ts_placement_study/run_scan.py --laser-through-axis --interferometer-only   # 15 rows, best 14.03 bit
python $P/run_placement_inference.py --study if_and_ts_placement_study --workers 4
python $P/run_placement_inference.py --study if_and_ts_placement_study \
    --layouts initial_guess optimized separately_optimized sequentially_optimized \
    --n-seeds 5 --skip-stored --workers 4
python $P/run_placement_inference.py --study if_and_ts_placement_study \
    --layouts initial_guess optimized \
    --truths test_state_1_parametric test_state_2_parametric test_state_3_parametric \
    test_state_4_parametric test_state_5_gp test_state_6_gp test_state_7_gp test_state_8_gp --workers 4
```

Notebook: `results_plots_tables/results/diagnostics_placement_optimization/if_and_ts_placement_study/results.ipynb`
