# data

This directory contains some summary data from the main analyses run in our manuscript.

- [`./raw/neurosynth/terms.txt`](./raw/neurosynth/terms.txt): A text file with the 123 terms used to run the meta-analyses reported in the manuscript.
  These terms were the overlap of terms in the NeuroSynth database and those listed in the [Cognitive Atlas](https://www.cognitiveatlas.org/).
- [`./derivatives/simulated/pval_summary.csv.gz`](./derivatives/simulated/pval_summary.csv.gz): A CSV file with p-values of the null models generated from the simulation analyses examining correlated (r = 0.15) brain maps.
- [`./derivatives/simulated/prob_summary.csv.gz`](./derivatives/simulated/prob_summary.csv.gz): A CSV file with the false positive rates of the various null models from the simulation analyses.
- [`./derivatives/simulated/moran_summary.csv.gz`](./derivatives/simulated/moran_summary.csv.gz): A CSV file with the Moran's I statistic of the empirical and null maps generated from the various null frameworks for the simulation analyses.
- [`./derivatives/neurosynth/ns_summary.csv.gz`](./derivatives/neurosynth/summary.csv.gz): A CSV file with the summary statistics of the primary NeuroSynth analyses (i.e., the number of significant correlations remaining after thresholding using each null framework).
- [`./derivatives/hcp/summary_thresh100.csv.gz`](./derivatives/hcp/summary_thresh100.csv.gz): A CSV file with the summary statistics of the primamry HCP analyses (i.e., the network-specific T1w/T2w z-scores and p-values after normalizing using each null framework).

We provide these CSV files for the sake of reproducibility; note, however, that all of these results can be generated programmatically.
(Refer to our [walkthrough](https://netneurolab.github.io/markello_spatialnulls) for instructions on how to do this.)
