# Spatial null models

The different spatially-constrained null model implementations are the crux of our manuscript.
Many of them already had easy-to-use Python interfaces (i.e., [Burt-2020](https://brainsmash.readthedocs.io/) and [Moran](https://brainspace.readthedocs.io/)), while others were easy to translate from open-source MATLAB or R code (i.e., [Váša](https://github.com/frantisekvasa/rotate_parcellation)).

Because #reasons, some of the null models were translated / implemented in our lab's catch-all utility package [`netneurotools`](https://github.com/netneurolab/netneurotools) (which you will find is used quite heavily throughout our analysis scripts), while other models (namely, Burt-2018) were implemented in a little helper package developed for this project ([`parspin`](https://github.com/netneurolab/markello_spatialnulls/blob/master/parspin)).
Here, we provide a brief code snippet highlighting the implementation for each null models used.
We quote relevant text from the manuscript describing the null models.

```{note}
When `brain` is used in the code snippets below it references a vector of length (N,) representing the original brain map that is being tested.
When `distance` is used in the code snippets below it references an array of shape (N, N) representing a geodesic distance matrix.
The `n_perm` variable is the number of permutations / spins / surrogates to be generated (1,000 for simulations, 10,000 for empirical datasets).
```

## Naive models

### Parametric

> Although the exact implementation of the parametric method varies based on the statistical test employed, all implementations share a reliance on standard null distributions.
> For example, when examining correlation values, the parametric method relies on the Student's *t*-distribution; when examining z-statistics, this method uses the standard normal distribution.

```python
from scipy import stats

# map correspondence
rcorr, p_value = stats.pearsonr(brain1, brain2)

# partition specificity
tstat, p_value = stats.ttest_1samp(brain[network], 0)
```

### Non-parametric

> "The naive non-parametric approach uses a random permutation (i.e., reshuffling) of the data to construct a null distribution, destroying its inherent spatial structure.
> Each vertex or parcel is uniquely reassigned the value of another vertex or parcel for every permutation."

```python
from numpy import np

rs = np.random.default_rng(1234)
spins = np.column_stack([rs.permutation(len(brain)) for _ in range(n_perm)])
nulls = brain[spins]
```

## Spatial permutation models

### Vázquez-Rodríguez ([Source code](https://github.com/netneurolab/netneurotools/blob/a1beaed2ced9c5236f9635041e30aa1f037023eb/netneurotools/stats.py#L542))

> "The *Vázquez-Rodríguez* method, which serves as a direct adaptation of the original framework from Alexander-Bloch et al., 2018, *NeuroImage* but applied to parcellated brain data, was first used in Vázquez-Rodríguez et al., 2019, *PNAS*.
> In this adaptation, vertex coordinates are replaced with those of parcel centroids.
> That is, a rotation is applied to the coordinates for the center-of-mass of each parcel, and parcels are reassigned the value of the closest rotated parcel (i.e., that with the minimum Euclidean distance).
> If the medial wall is rotated into a region of cortex the value of the nearest parcel is assigned instead, ensuring that all parcels have values for every rotation.
> This method for handling the medial wall consequently permits the duplicate reassignment of parcel values for every rotation, such that some parcel values may not be present in a given rotation and others may appear more than once.
> Note that the exact method used to define parcel centroids may impact the performance of this model."

#### Vertex data

```python
from netneurotools import freesurfer, stats

coords, hemi = freesurfer.get_fsaverage_coords('fsaverage5', 'sphere')
spins = stats.gen_spinsamples(coords, hemi, seed=1234)
nulls = brain[spins]
```

#### Parcellated data

```python
from netneurotools import datasets, freesurfer, stats

annot = datasets.fetch_cammoun2012('fsaverage5')['scale500']
coords, hemi = freesurfer.find_parcel_centroids(lhannot=annot.lh,
                                                rhannot=annot.rh,
                                                version='fsaverage5',
                                                surf='sphere',
                                                method='surface')
spins = stats.gen_spinsamples(coords, hemi, seed=1234)
nulls = brain[spins]
```

### Baum ([Source code](https://github.com/netneurolab/netneurotools/blob/a1beaed2ced9c5236f9635041e30aa1f037023eb/netneurotools/freesurfer.py#L568))

> "Used initially in Baum et al., 2020, *PNAS*, this method projects parcellated brain data to a high-resolution surface mesh, assigning identical values to all the vertices within a given parcel.
> The projected mesh is subjected to the original spatial permutation reassignment procedure Alexander-Bloch et al., 2018, *NeuroImage* and re-parcellated by taking the modal (i.e., the most common) value of the vertices in each parcel.
> When the rotated medial wall completely subsumes a cortical parcel that region is assigned a value of NaN and is removed from subsequent analyses.
> Notably, this method can result in duplicate assignment of parcel values in each permutation."

```python
from netneurotools import datasets, freesurfer

annot = datasets.fetch_cammoun2012('fsaverage5')['scale500']
spins = freesurfer.spin_parcels(lhannot=annot.lh, rhannot=annot.rh,
                                version='fsaverage5', seed=1234)
nulls = brain[spins]
nulls[spins == -1] = np.nan
```

### Cornblath ([Source code](https://github.com/netneurolab/netneurotools/blob/a1beaed2ced9c5236f9635041e30aa1f037023eb/netneurotools/freesurfer.py#L482))

> "In this method implemented by Cornblath et al., *Commun Bio* parcellated data are projected to a high-resolution spherical surface mesh, rotated, and re-parcellated by taking the average (i.e., the arithmetic mean) of the vertices in each parcel.
> When the rotated medial wall completely subsumes a cortical parcel that region is assigned a value of NaN and is removed from subsequent analyses.
> Because the data are re-parcellated the likelihood of duplicated assignments is very low (though not exactly zero); however, the distribution of re-parcellated values will be slightly different than the original data distribution."

```python
from netneurotools import datasets, freesurfer

annot = datasets.fetch_cammoun2012('fsaverage5')['scale500']
nulls = freesurfer.spin_data(brain, lhannot=annot.lh, rhannot=annot.rh,
                             version='fsaverage5', seed=1234)
```

### Váša ([Source code](https://github.com/netneurolab/netneurotools/blob/a1beaed2ced9c5236f9635041e30aa1f037023eb/netneurotools/stats.py#L542))

> "The first known application of spatial permutations to parcellated data, the *Váša* method (Váša et al., 2018, *Cereb Cortex*) attempted to resolve one of the primary drawbacks of the Alexander-Bloch method: duplicate reassignment of values.
>That is, this method was created so as to yield a "perfect" permutation of the original data for every rotation.
> Similar to the *Vázquez-Rodríguez* method, parcel centroids are used instead of vertex coordinates.
> In order to avoid duplicate reassignments, parcels are iteratively assigned by (1) finding the closest rotated parcel to each original parcel, and (2) assigning the most distant pair of parcels.
> This two-step process is then repeated for all remaining unassigned parcels until each has been reassigned.
>Parcels are reassigned without consideration for the medial wall or its rotated location.
> Note that the exact method used to define parcel centroids may impact the performance of this model."

```python
from netneurotools import datasets, freesurfer, stats

annot = datasets.fetch_cammoun2012('fsaverage5')['scale500']
coords, hemi = freesurfer.find_parcel_centroids(lhannot=annot.lh,
                                                rhannot=annot.rh,
                                                version='fsaverage5',
                                                surf='sphere',
                                                method='surface')
spins = stats.gen_spinsamples(coords, hemi, method='vasa', seed=1234)
nulls = brain[spins]
```

### Hungarian ([Source code](https://github.com/netneurolab/netneurotools/blob/a1beaed2ced9c5236f9635041e30aa1f037023eb/netneurotools/stats.py#L542))

> "Similar to the *Váša* method, the *Hungarian* method attempts to uniquely reassign each parcel for every rotation.
> Instead of using an iterative process, however, which can result in globally sub-optimal assignments, this method uses the Hungarian algorithm to solve a linear sum assignment problem (Kuhn, 1955, *Nav Res Logist Q*).
> This method attempts to uniquely reassign each parcel such that the global reassignment cost is minimized, where cost is quantified as the distance between the original and rotated parcel centroid coordinates.
> The medial wall is ignored in all rotations and the optimal reassignment is determined without consideration for its location.
> Note that the exact method used to define parcel centroids may impact the performance of this model."

```python
from netneurotools import datasets, freesurfer, stats

annot = datasets.fetch_cammoun2012('fsaverage5')['scale500']
coords, hemi = freesurfer.find_parcel_centroids(lhannot=annot.lh,
                                                rhannot=annot.rh,
                                                version='fsaverage5',
                                                surf='sphere',
                                                method='surface')
spins = stats.gen_spinsamples(coords, hemi, method='hungarian', seed=1234)
nulls = brain[spins]
```

## Parameterized data models

### Burt-2018 ([Source code](https://github.com/netneurolab/markello_spatialnulls/blob/master/parspin/parspin/burt.py#L88))

The Burt-2018 model must be run separately on each hemisphere; we show the reference call below assuming data from a single hemisphere.
Note also that the `brain` data provided to the Burt-2018 method must be *positive*.

> "Described in Burt et al., 2018, *Nat Neuro*, this framework uses a spatial autoregressive model of the form $\mathbf{y} = \rho \mathbf{W} \mathbf{y}$ to generate surrogate data.
> Here, $\mathbf{y}$ refers to a Box-Cox transformed, mean-centered brain feature of interest (i.e., a brain map), $\mathbf{W}$ is a weight matrix (derived from $\mathbf{D}$, a matrix of the distance between brain regions, and $d_{0}$, a spatial autocorrelation factor), and $\rho$ is a spatial lag parameter.
> The parameters $\rho$ and $d_{0}$ are derived from the data via a least-squares optimization procedure and their estimates $\hat{\rho}$ and $\hat{d_{0}}$ are used to generate surrogate brain maps according to $\mathbf{y_{surr}} = (\mathbb{I} - \hat{\rho} \mathbf{W}[\hat{d_{0}}])^{-1} \mathbf{u}$, where $\mathbf{u} \sim \mathcal{N}(0,1)$ is a vector of random Gaussian noise.
> Rank-ordered values in the $\mathbf{y_{surr}}$ map are replaced with corresponding values from the original $\mathbf{y}$."

```python
from parspin import burt

nulls = burt.batch_surrogates(brain, distance, seed=1234)
```

### Burt-2020

The Burt-2020 model must be run separately on each hemisphere; we show the reference call below assuming data from a single hemisphere.

> "Two years after introducing their spatial autoregressive method, Burt et al., 2020, *NeuroImage* proposed a novel model to generate surrogate data using variogram estimation.
> The method operates in two main steps: (1) randomly permute the values in a given brain map, and (2) smooth and re-scale the permuted values to reintroduce spatial autocorrelation characteristic of the original, non-permuted data.
> Reintroduction of spatial autocorrelation onto the permuted data is achieved via the transformation $\mathbf{y} = |\beta|^{1/2} \mathbf{x'} + |\alpha|^{1/2} \mathbf{z}$, where $\mathbf{x'}$ is the permuted data, $\mathbf{z} \sim \mathcal{N}(0,1)$ is a vector of random Gaussian noise, and $\alpha$ and $\beta$ are estimated via a least-squares optimization between variograms of the original and permuted data.
> When applied to empirical data, rank-ordered values in the surrogate map are replaced with corresponding values from the original brain map; surrogates maps generated from simulated data use the raw values of $\mathbf{y}$."

#### Vertex data ([Source code](https://github.com/murraylab/brainsmash/blob/f45ee62570fc0d8e764f40cfa90de9086780ab19/brainsmash/mapgen/base.py#L16))

```python
from brainsmash import mapgen

index = np.argsort(distance, axis=-1)
distance = np.sort(distance, axis=-1)
nulls = brainsmash.mapgen.Sampled(brain, distance, index, seed=1234)(n_perm).T
```

#### Parcellated data ([Source code](https://github.com/murraylab/brainsmash/blob/f45ee62570fc0d8e764f40cfa90de9086780ab19/brainsmash/mapgen/sampled.py#L16))

```python
from brainsmash import mapgen

nulls = mapgen.Base(brain, distance, seed=1234)(n_perm).T
```

Note: for analyses of **empirical** data in our manuscript we also provided the `resample=True` parameter.
Surrogates maps for simulated data were always generated with `resample=False` (default).

### Moran ([Source code](https://github.com/MICA-MNI/BrainSpace/blob/1fb001f4961d3c0b05b7715f42bcc362b31b96a5/brainspace/null_models/moran.py#L215))

The Moran model must be run separately on each hemisphere; we show the reference call below assuming data from a single hemisphere.

> "Originally developed in the ecology literature (Dray et al., 2011, *Geogr Anal*; Wagner et al., 2015, *Methods Ecol Evol*), *Moran spectral randomization* (MSR) has only been recently applied to neuroimaging data (Paquola et al., 2020, *PLoS Biol*, vos de Wael et al., *Comm Bio*, Royer et al., 2020, *NeuroImage*).
> Similar to the other parameterized data methods, MSR principally relies on a spatially-informed weight matrix $\mathbf{W}$, usually taking the form of an inverse distance matrix between brain regions.
> However, rather than using $\mathbf{W}$ to estimate parameters via a least-squares approach, MSR uses an eigendecomposition of $\mathbf{W}$ to compute spatial eigenvectors that provide an estimate of autocorrelation.
> These eigenvectors are then used to impose a similar spatial structure on random, normally distributed surrogate data."

```python
from brainspace import null_models

np.fill_diagonal(dist, 1)
dist **= -1
mrs = null_models.MoranRandomization(joint=True, tol=1e-6, random_state=1234)
nulls = mrs.fit(distance).randomize(brain).T
```
