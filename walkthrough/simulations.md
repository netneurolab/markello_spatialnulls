# Simulated data analyses

Adapting methodology from Burt et al., 2020, *NeuroImage*, we ran two comprehensive simulation analyses to assess how accurate the different null frameworks are when there is a known "ground truth."
Here, we briefly describe the generation of the simulated brain maps and associated analyses used to generate results shown in the manuscript.

Note that all simulation analyses can be run with the `make simulations` command run from the root of the [repository](https://github.com/netneurolab/markello_spatialnulls).
Visualization of empirical results can be run with the `make plot_empirical` command, and supplementary analyses can be run with the `make suppl_empirical` command.

```{warning}
Although we provide the `make simulations` command to be consistent with other aspects of the data processing and analysis, we **strongly** encourage you not to run the simulation analyses this way.
The simulation analyses are very computationally-/time-intensive, and we estimate using `make simulations` will require *at least* 1 month of computation time.
[Below](#testing-the-null-models) we provide some batch scripts that can be used to run the simulations on an HPC cluster to significantly speed things up.
```

## Creating the simulated brain maps

The first step in this whole process is to create simulated brain maps.
Critically, we need to (1) be able to control the degree of spatial autocorrelation in the maps, (2) project the maps to the cortical surface (for use with spatial permutation null methods), and (3) specify the correlation between pairs of maps.

To solve the first problem—controlling spatial autocorrelation—we use code generously provided by Burt and colleagues (2020, *NeuroImage*).

To solve the second problem—projecting the maps to the cortical surface—we adapt this code, which was used to generate 2-dimensional Gaussian random fields, to make 3D fields instead.
We then pretend hese 3D fields are simple volumetric brain images, and use FreeSurfer's `mri_vol2surf` to get cortical representations of the maps.

The final problem—specifying the correlation between pairs of maps—is a bit trickier.
We can try and brute force it, generating random maps over and over until we get two that have a set correlations, and while this will succeed quite readily when the amount of spatial autocorrelation in the maps is high, at low levels of spatial autocorrelation (or not spatial autocorrelation!) achieving our target correlation value of 0.15 would be nigh impossible.
As such, we use a multivariate normal distribution to create two random vectors that are correlated to our target *r*, and then use the above procedure to convert them into spatially-autocorrelated brain maps on the surface.
If the generated brain maps remain correlated to our target *r* we use them in the rest of the analysis, and if they don't we discard and try again with a different set of correlated random vectors.

Code for generating simulated brain pairs can be found in [`scripts/simulations/generate_simulations.py`](https://github.com/netneurolab/markello_spatialnulls/blob/master/scripts/simulations/generate_simulations.py).

## Testing the null models

You might notice that there are two scripts in the `scripts/simulations` directory that seem wildly similar ([`scripts/simulations/run_simulated_nulls_serial.py`](https://github.com/netneurolab/markello_spatialnulls/blob/master/scripts/simulations/run_simulated_nulls_serial.py) and [`scripts/simulations/run_simulated_nulls_parallel.py`](https://github.com/netneurolab/markello_spatialnulls/blob/master/scripts/simulations/run_simulated_nulls_parallel.py)), and you'd be right: the scripts are *nearly* identical.
Their primary difference is in what level we are parallelizing the computation of the null maps.

The `run_simulated_nulls_parallel.py` script applies the null models to the simulated data, running different *simulations* in parallel (that is, there is no parallelization at the level of null map generation).
This is most useful for null frameworks where running the simulation is the most costly computational step (e.g., for the *Vázquez-Rodríguez*, *Baum*, *Cornblath*, *Váša*, *Hungarian*, *Moran*, and *naive non-parametric* methods).
On the other hand, `run_simulated_nulls_serial.py` script applies the null models to the simulated data, where the *generation of null maps* is done in parallel (that is, the simulations are run serially).
This is most useful for null frameworks where generating the null maps for a given method is the most costly computational step (i.e., for the *Burt-2018* and *Burt-2020* methods).

### Running null models "in serial"

Below we provide the outline of a batch script for submission to a SLURM job scheduler on an HPC for use with null models that are best run "in serial" (i.e., the *Burt-2018* and *Burt-2020* models).
Because these models take so long to run, we recommend using array jobs to split them up into smaller chunks that can be run in parallel (and also running the different spatial autocorrelation levels in parallel, too).
Note that in the below script you will have to specify the `$ALPHA` and `$SPATNULL` variable, and submit a separate job for each of these variable combinations.
(You could change these variables to parameters and use a separate script to iterate through and submit the different combinations.)

The time, CPU, and memory requirements can be modified as you see fit, but we find the following combinations work moderately well (and should avoid memory errors):

- *Burt-2018*: 3 days, 24 CPUs, 48G
- *Burt-2020*: 3 days, 24 CPUs, 96G

```bash
#!/bin/bash
#SBATCH --time=3-00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --array=0-950:50

# which null model / spatial autocorrelation combo are we running?
ALPHA=alpha-0.0
SPATNULL=burt2020

# run the script 
SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_MKL_NUM_THREADS=1 \
singularity run --cleanenv -B ${PWD}:/opt/spatnulls --pwd /opt/spatnulls \
                ${PWD}/container/markello_spatialnulls.simg \
                python scripts/simulations/run_simulated_nulls_serial.py \
                --n_perm 10000 --n_proc ${SLURM_CPUS_PER_TASK} --seed 1234 \
                --alpha ${ALPHA} --spatnull ${SPATNULL} \
                -- ${SLURM_ARRAY_TASK_ID} 50
```

You'd then need to iterate over all the spatial autocorrelation parameters (`alpha-0.0` through `alpha-3.0`) and the different null models you wanted to run this way (`burt2018`, `burt2020`) and re-submit jobs for each (and then watch as your HPC allocation priority vanishes into thin air as your jobs queue up and process!).

### Running null models "in parallel"

Below we provide the outline of a batch script for submission to a SLURM job scheduler on an HPC for use with null models that are best run "in parallel" (i.e., the *Vázquez-Rodríguez*, *Baum*, *Cornblath*, *Váša*, *Hungarian*, *Moran*, and *naive non-parametric* models).
Because these methods don't take nearly so long to run as those referenced above, we do not recommend using an array job for these.
Note, however, that you will still have to change the `$ALPHA` and `$SPATNULL` variables and submit a separate job for each of these variable combinations (or you can increase the time limit and run multiple at once by providing multiple frameworks to the `--spatnull` and `--alpha` parameters).

The time, CPU, and memory requirements can be modified as you see fit, but we find the following combinations work moderately well (and should avoid memory errors):

- *Vázquez-Rodríguez*, *Baum*, *Váša*, *Hungarian*, *naive non-parametric*: 3 hours, 12 CPUs, 24G
- *Cornblath*: 12 hours, 24 CPUs, 24G
- *Moran*: 24 hours, 12 CPUs, 24G

```bash
#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G

# which null model / spatial autocorrelation combo are we running?
ALPHA=alpha-0.0
SPATNULL=burt2020

# run the script 
SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_MKL_NUM_THREADS=1 \
singularity run --cleanenv -B ${PWD}:/opt/spatnulls --pwd /opt/spatnulls \
                ${PWD}/container/markello_spatialnulls.simg \
                python scripts/simulations/run_simulated_nulls_parallel.py \
                --n_perm 10000 --n_proc ${SLURM_CPUS_PER_TASK} --seed 1234 \
                --alpha ${ALPHA} --spatnull ${SPATNULL} -- 1000
```

## Visualizing simulation results

Once the simulations analyses have been generated the results can be plotted using the command `make plot_simulations`.
(Note that this assumes you have already run `make simulations`!)

This will save out a _lot_ of different individual plots, which we combined into the results shown in Figures 2, 3 and S1-5 in the manuscript.

## Supplementary analyses

We tested a few extra things in our analyses to see how minor variations in implementation of different null models impacted the results.
Namely, we used simulated data to assess the [relative runtimes](https://github.com/netneurolab/markello_spatialnulls/blob/master/scripts/suppl_simulations/compare_comptime.py) of the different null models, and how the [size of the null distribution](https://github.com/netneurolab/markello_spatialnulls/blob/master/scripts/suppl_simulations/compare_nnulls.py) created by each model impacts the accuracy of the statistical estimates they generate.

You can generate these supplementary results with the command `make suppl_simulations`.
(Note that this assumes you have already run `make simulations`!)
