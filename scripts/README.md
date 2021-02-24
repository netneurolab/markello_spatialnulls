# scripts

This directory contains the analytic scripts comprising the backbone of the manuscript.
If you read something in the manuscript and have a question about the methodology or implementation chances are you can find the answer in one of these files.

## `simulations`


- [`generate_simulations.py`](./simulations/generate_simulations.py):
  Generates simulated Gaussian random fields on the fsaverage5 surface across seven levels of spatial autocorrelation.
  Data are also parcellated with the multi-scale Cammoun and Schaefer atlases.
  Generated maps are saved to `data/derivatives/simulated/{alpha}/sim`, where {alpha} indicates the level of spatial autocorrelation.
- [`run_simulated_nulls_parallel.py`](./simulations/run_simulated_nulls_parallel.py):
  Runs null models on simulated data, where different *simulations* are run in parallel (that is, there is no parallelization at the level of each null model).
  This is most useful for null frameworks where running the simulation is the most costly computational step (e.g., Vázquez-Rodríguez, Baum, Cornblath, Váša, Hungarian, naive non-parametric)
  By default this will generate p-values for the correlated simulations (for calculating the -log10(p) of the frameworks).
  When run with the `--shuffle` flag it will generate p-values for the randomized simulations (for calculating the false positive rate)
  Data are saved to `data/derivatives/simulated`.
- [`run_simulated_nulls_serial.py`](./simulations/run_simulated_nulls_serial.py):
  Runs null models on simulated data, where *null models* are run in parallel (that is, there is no parallelization at the level of simulations).
  This is most useful for null frameworks where generating the null maps is the most costly computational step (e.g., Burt-2018, Burt-2020, Moran).
  By default this will generate p-values for the correlated simulations (for calculating the -log10(p) of the frameworks).
  When run with the `--shuffle` flag it will generate p-values for the randomized simulations (for calculating the false positive rate)
  Data are saved to `data/derivatives/simulated`.
- [`combine_simnulls_outputs.py`](./simulations/combine_simnulls_outputs.py):
  Combines all the outputs from `run_simulated_nulls_parallel.py` and `run_simulated_nulls_serial.py` into three files: `prob_summary.csv.gz`, `pval_summary.csv.gz`, and `auc_summary.csv.gz`.
  Data are saved to `data/derivatives/simulated`
- [`combine_moran_outputs.py`](./simulations/combine_moran_outputs.py):
  Combines the outputs generated from running `run_simulated_nulls_serial.py` with the `--run_moran` flag in a single file: `moran_summary.csv.gz`.
  Data are saved to `data/deriatives/simulated`.

You can run these scripts from the root of the repository using the command `make simulations`.

## `empirical`

- [`fetch_neurosynth_maps.py`](./empirical/fetch_neurosynth_maps.py):
  Downloads association maps from NeuroSynth and projects them to the fsaverage5 surface (requires FreeSurfer).
  Only download association maps for NeuroSynth terms overlapping with those in the Cognitive Atlas.
  Raw maps are stored to `data/raw/neurosynth` and parcellated data are stored in `data/derivatives/neurosynth`
- [`fetch_hcp_myelin.py`](./empirical/fetch_hcp_myelin.py):
  Splits group-averaged HCP myelin map into left/right hemisphere and resamples to the fsaverage5 surface (requires Connectome Workbench).
  Raw data are stored in `data/raw/hcp` and parcellated data are stored in `data/derivatives/hcp`
- [`get_geodesic_distance.py`](./empirical/get_geodesic_distance.py):
  Calculates parcel-parcel geodesic distance matrices fro all resolutions of the Cammoun + Schaefer atlases.
  Here, parcel-parcel distance is operationalized as the average of the distance between pairs of vertices in both parcels.
  Distance matrices are saved to `data/derivatives/geodesic`.
- [`generate_spin_resamples.py`](./empirical/generate_spin_resamples.py):
  Generates resampling arrays for spatial permutation null models.
  Resampling arrays are stored in `data/derivatives/spins`.
- [`generate_neurosynth_surrogates.py`](./empirical/generate_neurosynth_surrogates.py):
  Generates parameterized data surrogates for NeuroSynth maps.
  Data are stored in `data/derivatives/surrogates/<atlas>/<null_method>/neurosynth`.
- [`generate_hcp_surrogates.py`](./empirical/generate_hcp_surrogates.py):
  Generates paramaterized data surrogates for HCP T1w/T2w map.
  Data are stored in `data/derivatives/surrogates/<atlas>/<null_method>/hcp`.
- [`run_neurosynth_nulls.py`](./empirical/run_neurosynth_nulls.py):
  Generates statistical estimates for NeuroSynth analyses using all null models.
  Output summary file is saved to `data/derivatives/neurosynth/summary.csv`.
- [`run_hcp_nulls.py`](./empirical/run_hcp_nulls.py):
  Generates statistical estimates for HCP analyses using all null models.
  Output summary file is saved to `data/derivatives/hcp/summary.csv`.

You can run these scripts from the root of the repository using the command `make empirical`.

## `plot_simulations`

- [`viz_simulation_examples.py`](./plot_simulations/viz_simulation_examples.py):
  Creates example simulation brain maps from the same underlying data with varying levels of spatial autocorrelation.
  Files are saved to `figures/simulated/examples`.
- [`viz_simulation_results.py`](./plot_simulations/viz_simulation_results.py):
  Creates figures for visualizing the simulation analyses.
  Files are saved to `figures/simulated`.

You can run these scripts from the root of the repository using the command `make plot_simulations`.

## `plot_empirical`

- [`viz_perms.py`](./plot_empirical/viz_perms.py):
  Saves example image of one rotation / surrogate for each null model.
  Files are saved to `figures/spins/examples`.
- [`viz_neurosynth_analysis.py`](./plot_empirical/viz_neurosynth_analysis.py):
  Creates figures for visualizing the analytic procedure of the NeuroSynth analysis.
  Files are saved to `figures/neurosynth/analysis`.
- [`viz_neurosynth_nulls.py`](./plot_empirical/viz_neurosynth_nulls.py):
  Saves figures from the NeuroSynth analyses for all null models.
  Files are saved to `figures/neurosynth`.
- [`viz_hcp_analysis.py`](./plot_empirical/viz_hcp_analysis.py):
  Creates figures for visualizing the analytic procedure of the HCP analysis.
  Files are saved to `figures/hcp/analysis`.
- [`viz_hcp_nulls.py`](./plot_empirical/viz_hcp_nulls.py):
  Saves figures from the HCP analyses for all null models.
  Files are saved to `figures/hcp`.
- [`viz_hcp_networks.py`](./plot_empirical/viz_hcp_networks.py):
  Saves supplementary figure for visualizing null distributions from the HCP analysis.
  Files are saved to `figures/hcp`.

You can run these scripts from the root of the repository using the command `make plot_empirical`.

## `suppl_simulations`

- [`compare_comptime.py`](./suppl_simulations/compare_comptime.py):
  Assesses the runtime of the different null models, comparing how caching of intermediate data saves on computation.
  Output figures are saved to `figures/supplementary/comp_time`.
- [`compare_nnulls.py`](./suppl_simulations/compare_nnulls.py):
  Assesses how the size of the null distribution generated by each null framework influences the resulting statistical estimate.
  Output figures are saved to `figures/supplementary/comp_nnulls`.

You can run these scripts from the root of the repository using the command `make suppl_simulations`.

## `suppl_empirical`

- [`compare_spin_resamples.py`](./suppl_empirical/compare_spin_resamples.py):
  Assesses how parcel centroid definition impacts the resampling arrays generated from three spatial permutation null models (i.e., Vazquez-Rodriguez, Vasa, and Hungarian).
  Output figures are saved to `figures/supplementary/comp_spins`.
- [`compare_geodesic_travel.py`](./suppl_empirical/compare_geodesic_travel.py):
  Assesses how (dis)-allowing travel along the medial wall when constructing parcel-parcel geodesic distance matrices impacts the surrogates generated from parameterized data null models.
  Output figures are saved to `figures/supplementary/comp_geodesic`.

You can run these scripts from the root of the repository using the command `make suppl_empirical`.
