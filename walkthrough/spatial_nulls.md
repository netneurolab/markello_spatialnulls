# Spatial null models

The different spatially-constrained null model implementations are the crux of our manuscript.
Thankfully, some of them already had easy-to-use Python interfaces (i.e., [Burt-2020](https://brainsmash.readthedocs.io/) and [Moran](https://brainspace.readthedocs.io/)), while others had open MATLAB or R code (i.e., [Váša](https://github.com/frantisekvasa/rotate_parcellation)) that needed to be translated.

To generate all the spatial nulls you can use the command `make analysis` from the root of the [repository](https://github.com/netneurolab/markello_spatialnulls).
(Note that this assumes you have already run `make preprocess` from the [last step](./preprocessing.md)!)

## All the Python implementations

Because reasons, some of the null models were translated / implemented in our lab's catch-all utility package [`netneurotools`](https://github.com/netneurolab/netneurotools), which you will find is used quite heavily throughout our analysis scripts, while other models (namely, Burt-2018) were implemented in a little helper package developed for this project ([`parspin`](https://github.com/netneurolab/markello_spatialnulls/blob/master/parspin)).
Here, we point to the implementation used for each of the null models so that curious researchers know where to look.

(Refer to the following scripts to see how these frameworks were used in the current analyses: (1) [`scripts/02_analysis/generate_spin_resamples.py`](https://github.com/netneurolab/markello_spatialnulls/blob/master/scripts/02_analysis/generate_spin_resamples.py), (2) [`scripts/02_analysis/generate_hcp_surrogates.py`](https://github.com/netneurolab/markello_spatialnulls/blob/master/scripts/02_analysis/generate_hcp_surrogates.py), and (3) [`scripts/02_analysis/generate_neurosynth_surrogates.py`](https://github.com/netneurolab/markello_spatialnulls/blob/master/scripts/02_analysis/generate_neurosynth_surrogates.py).)

### Vázquez-Rodríguez

Usage: `netneurotools.stats.gen_spinsamples(coords, hemi, method='original', seed=1234)`

[Source code](https://github.com/netneurolab/netneurotools/blob/master/netneurotools/stats.py#L513)

Note: so-called "original" because it was a direct adaptation of spin-based resampling from the "original" Alexander-Bloch et al., 2018, *NeuroImage* method.

### Váša

Usage: `netneurotools.stats.gen_spinsamples(coords, hemi, method='vasa', seed=1234)`

[Source code](https://github.com/netneurolab/netneurotools/blob/master/netneurotools/stats.py#L513)

### Hungarian

Usage: `netneurotools.stats.gen_spinsamples(coords, hemi, method='hungarian')`

[Source code](https://github.com/netneurolab/netneurotools/blob/master/netneurotools/stats.py#L513)

Note: this algorithm is _very_ slow for higher-resolution parcellations.

### Baum

Usage: `netneurotools.freesurfer.spin_parcels(lhannot, rhannot, version='fsaverage5', seed=1234)`

[Source code](https://github.com/netneurolab/netneurotools/blob/master/netneurotools/freesurfer.py#L573)

### Cornblath

Usage: `netneurotools.freesurfer.spin_data(brain, lhannot, rhannot, version='fsaverage5', seed=1234)`

[Source code](https://github.com/netneurolab/netneurotools/blob/master/netneurotools/freesurfer.py#L487)

### Burt-2018

Usage: `parspin.burt.make_surrogate(distance, brain, seed=1234)`

[Source code](https://github.com/netneurolab/markello_spatialnulls/blob/master/parspin/parspin/burt.py#L86)

### Burt-2020

Usage: `brainsmash.mapgen.Base(brain, distance, resample=True, seed=1234)`

[Source code](https://github.com/murraylab/brainsmash/blob/master/brainsmash/mapgen/base.py#L15)

Note: we used `resample=True` throughout the manuscript despite the [original authors' warnings](https://brainsmash.readthedocs.io/en/latest/source/brainsmash.mapgen.html#brainsmash.mapgen.base.Base) that this may cause the variograms of the surrogate maps to be slightly less well-matched to the original data.
The reason for this was computational: using `resample=True` allowed us to store the resampling arrays as integers (rather than storing the surrogates as floats).

### Moran

Usage: `brainspace.null_models.MoranRandomization(joint=True, tol=1e-6, random_state=1234)`

[Source code](https://github.com/MICA-MNI/BrainSpace/blob/master/brainspace/null_models/moran.py#L215)
