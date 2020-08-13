# Null framework performance

Now that the data are pre-processed and the null models are (mostly) pre-generated, we can test the performance of the nulls on some actual analyses!

To generate the primary results you can use the command `make results` from the root of the [repository](https://github.com/netneurolab/markello_spatialnulls).
(Note that this assumes you have already run `make preprocess` and `make analysis`!)

## Testing brain map correspondence (NeuroSynth)

All code for running the brain map correspondence analyses can be found in [`scripts/03_results/run_neurosynth_nulls.py`](https://github.com/netneurolab/markello_spatialnulls/blob/master/scripts/03_results/run_neurosynth_nulls.py).

## Testing partition specificity (HCP)

All code for running the partition specificity analyses can be found in [`scripts/03_results/run_hcp_nulls.py`](https://github.com/netneurolab/markello_spatialnulls/blob/master/scripts/03_results/run_hcp_nulls.py).

## Supplementary analyses

We tested a few extra things in our analyses to see how minor variations in implementation of different null models impacted the results.
You can generate these supplementary results with the command `make supplementary` from the root of the [repository](https://github.com/netneurolab/markello_spatialnulls).
(Note that this assumes you have already run `make preprocess` and `make analysis`!)
