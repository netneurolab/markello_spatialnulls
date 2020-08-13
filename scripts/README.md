# scripts

This directory contains the analytic scripts comprising the backbone of the manuscript. If you read something in the manuscript and have a question about the methodology or implementation chances are you can find the answer in one of these files.

## 01_preprocess

- [fetch_neurosynth_maps.py](./01_preprocess/fetch_neurosynth_maps.py):
  Downloads association maps from NeuroSynth and projects them to the fsaverage5 surface (requires FreeSurfer).
  Only download association maps for NeuroSynth terms overlapping with those in the Cognitive Atlas.
  Raw maps are stored to `data/raw/neurosynth` and parcellated data are stored in `data/derivatives/neurosynth`
- [fetch_hcp_myelin.py](./01_preprocess/fetch_hcp_myelin.py):
  Splits group-averaged HCP myelin map into left/right hemisphere and resamples to the fsaverage5 surface (requires Connectome Workbench).
  Daw data are stored in `data/raw/hcp` and parcellated data are stored in `data/derivatives/hcp`

You can run these scripts from the root of the repository using the command `make preprocess`.

## 02_analysis

- [get_geodesic_distance.py](./02_analysis/get_geodesic_distance.py):
  Calculates parcel-parcel geodesic distance matrices fro all resolutions of the Cammoun + Schaefer atlases.
  Here, parcel-parcel distance is operationalized as the average of the distance between pairs of vertices in both parcels.
  Distance matrices are saved to `data/derivatives/geodesic`.
- [generate_spin_resamples.py](./02_analysis/generate_spin_resamples.py):
  Generates resampling arrays for spatial permutation null models.
  Resampling arrays are stored in `data/derivatives/spins`.
- [generate_neurosynth_surrogates.py](./02_analysis/generate_neurosynth_surrogates.py):
  Generates parameterized data surrogates for NeuroSynth maps.
  Data are stored in `data/derivatives/surrogates/<atlas>/<null_method>/neurosynth`.
- [generate_hcp_surrogates.py](./02_analysis/generate_hcp_surrogates.py):
  Generates paramaterized data surrogates for HCP T1w/T2w map.
  Data are stored in `data/derivatives/surrogates/<atlas>/<null_method>/hcp`.

You can run these scripts from the root of the repository using the command `make analysis`.

## 03_results

- [run_neurosynth_nulls.py](./03_results/run_neurosynth_nulls.py):
  Generates statistical estimates for NeuroSynth analyses using all null models.
  Output summary file is saved to `data/derivatives/neurosynth/summary.csv`.
- [run_hcp_nulls.py](./03_results/run_hcp_nulls.py):
  Generates statistical estimates for HCP analyses using all null models.
  Output summary file is saved to `data/derivatives/hcp/summary.csv`.

You can run these scripts from the root of the repository using the command `make results`.

## 04_visualization

- [viz_perms.py](./04_visualization/viz_perms.py):
  Saves example image of one rotation / surrogate for each null model.
  Files are saved to `figures/spins/examples`.
- [viz_neurosynth_nulls.py](./04_visualization/viz_neurosynth_nulls.py):
  Saves figures (box plots + line plots) from the NeuroSynth analyses for all null models.
  Files are saved to `figures/neurosynth`.
- [viz_hcp_nulls.py](./04_visualization/viz_hcp_nulls.py):
  Saves figures (bar graphs + heatmaps) from the HCP analyses for all null models.F
  Files are saved to `figures/hcp`.

You can run these scripts from the root of the repository using the command `make visualization`.

## 05_supplementary

- [compare_spin_resamples.py](./05_supplementary/compare_spin_resamples.py):
  Assesses how parcel centroid definition impacts the resampling arrays generated from three spatial permutation null models (i.e., Vazquez-Rodriguez, Vasa, and Hungarian).
  Output figures are saved to `figures/supplementary/comp_spins`.
- [compare_geodesic_travel.py](./05_supplementary/compare_geodesic_travel.py):
  Assesses how (dis)-allowing travel along the medial wall when constructing parcel-parcel geodesic distance matrices impacts the surrogates generated from parameterized data null models.
  Output figures are saved to `figures/supplementary/comp_geodesic`.

You can run these scripts from the root of the repository using the command `make supplementary`.
