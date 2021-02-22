# Simulated data analyses

Adapting methodology from Burt et al., 2020, *NeuroImage*, we ran two comprehensive simulation analyses to assess how accurate the different null frameworks are when there is a known "ground truth."
Here, we briefly describe the generation of the simulated brain maps and associated analyses used to generate results shown in the manuscript.

Note that all simulation analyses can be run with the `make simulations` command run from the root of the [repository](https://github.com/netneurolab/markello_spatialnulls)
Visualization of empirical results can be run with the `make plot_empirical` command, and supplementary analyses can be run with the `make suppl_empirical` command.

```{warning}
Although we provide the `make simulations` command to be consistent with other aspects of the data processing and analysis, we **strongly** encourage you not to run the simulation analyses this way.
The simulation analyses are very computationally- / time-intensive, and we estimate using `make simulations` will require *at least* 1 month of computation time.
Below we provide some batch scripts that can be used to run the simulations on an HPC cluster to significantly speed things up.
```

## Creating the simulated brain maps

## Testing the null models

### Running null models "serially"

### Running null models "in parallel"

## Visualizing simulation results

Once the simulations analyses have been generated the results can be plotted using the command `make plot_simulations`.
(Note that this assumes you have already run `make simulations`!)

This will save out a _lot_ of different individual plots, which we combined into the results shown in Figures 2, 3 and S1-5 in the manuscript.

## Supplementary analyses

We tested a few extra things in our analyses to see how minor variations in implementation of different null models impacted the results.
Namely, we used simulated data to assess the [relative runtimes](https://github.com/netneurolab/markello_spatialnulls/blob/master/scripts/suppl_simulations/compare_comptime.py) of the different null models, and how the [size of the null distribution](https://github.com/netneurolab/markello_spatialnulls/blob/master/scripts/suppl_simulations/compare_nnulls.py) created by each model impacts the accuracy of the statistical estimates they generate.

You can generate these supplementary results with the command `make suppl_simulations`.
(Note that this assumes you have already run `make simulations`!)
