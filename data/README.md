# data

This directory contains some summary data from the main analyses run in our manuscript.

- [./raw/neurosynth/terms.txt](./raw/neurosynth/terms.txt): A text file with the 123 terms used to run the meta-analyses reported in the manuscript.
  These terms were the overlap of terms in the NeuroSynth database and those listed in the [Cognitive Atlas](https://www.cognitiveatlas.org/).
- [./derivatives/neurosynth/summary.csv](./derivatives/neurosynth/summary.csv): A CSV file with the summary statistics of the primary NeuroSynth analyses (i.e., the number of significant correlations remaining after thresholding using each null framework).
- [./derivatives/hcp/summary.csv](./derivatives/hcp/summary.csv): A CSV file with the summary statistics of the primamry HCP analyses (i.e., the network-specific T1w/T2w z-scores and p-values after normalizing using each null framework).
  The `netclass` columns denotes whether the listed networks (in the `network` column) are from the Yeo intrinsic networks or the von Economo cytoarchitectonic classes.
