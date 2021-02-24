# -*- coding: utf-8 -*-
"""
Functions for working with + running null methods on simulated data
"""

from pathlib import Path

from joblib import Parallel, delayed, dump, load
import nibabel as nib
import numpy as np
import pandas as pd

from netneurotools import stats as nnstats
from parspin import spatial, utils

MAX_NSIM = 10000
MAX_NPERM = 10000
SPATNULLS = [  # all our null models
    'naive-para',
    'naive-nonpara',
    'vazquez-rodriguez',
    'baum',
    'cornblath',
    'vasa',
    'hungarian',
    'burt2018',
    'burt2020',
    'moran'
]
VERTEXWISE = [  # null models that can be run at vertex level
    'naive-para',
    'naive-nonpara',
    'vazquez-rodriguez',
    'burt2018',
    'burt2020',
    'moran'
]
ALPHAS = [  # available spatial autocorrelation params
    'alpha-0.0',
    'alpha-0.5',
    'alpha-1.0',
    'alpha-1.5',
    'alpha-2.0',
    'alpha-2.5',
    'alpha-3.0'
]


def load_spins(fn, n_perm=10000):
    """
    Loads spins from `fn`

    Parameters
    ----------
    fn : os.PathLike
        Filepath to file containing spins to load
    n_perm : int, optional
        Number of spins to retain (i.e., subset data)

    Returns
    -------
    spins : (N, P) array_like
        Loaded spins
    """

    if n_perm > MAX_NPERM:
        raise ValueError(f'Value for n_perm cannot exceed {MAX_NPERM}')

    npy = utils.pathify(fn).with_suffix('.npy')
    if npy.exists():
        spins = np.load(npy, allow_pickle=False, mmap_mode='c')[..., :n_perm]
    else:
        spins = np.loadtxt(fn, delimiter=',', dtype='int32')
        np.save(npy, spins, allow_pickle=False)
        spins = spins[..., :n_perm]

    return spins


def calc_pval(x, y, nulls):
    """
    Calculates p-values for simulations in `x` and `y` using `spatnull`

    Parameters
    ----------
    {x, y} : (N,) array_like
        Simulated GRF brain maps
    nulls : (N, P) array_like
        Null versions of `y` GRF brain map

    Returns
    -------
    pval : float
        P-value of correlation for `x` and `y` against `nulls`
    perms : np.ndarray
        Correlations of `x` with `nulls`
    """

    x, y, nulls = np.asanyarray(x), np.asanyarray(y), np.asanyarray(nulls)

    # calculate real + permuted correlation coefficients
    real = nnstats.efficient_pearsonr(x, y, nan_policy='omit')[0]
    perms = nnstats.efficient_pearsonr(x, nulls, nan_policy='omit')[0]
    pval = (np.sum(np.abs(perms) >= np.abs(real)) + 1) / (len(perms) + 1)

    return pval, perms


def load_parc_data(alphadir, parcellation, scale, sim=None, n_sim=MAX_NSIM):
    """
    Loads data for specified `parcellation`, `scale`, and `alpha`

    Parameters
    ----------
    alphadir : os.PathLike
        Filepath to directory for desired spatial autocorrelation nulls
    parcellation : {'atl-cammoun2012', 'atl-schaefer2018'}
        Name of parcellation to use
    scale : str
        Scale of parcellation to use. Must be valid scale for specified
        `parcellation`
    sim : {int, None}, optional
        Which simulation to load. If not specified, will load first `n_sim`
        simulations available. Default: None
    n_sim : int, optional
        Number of simulations to load, if `sim` is not specified. Default:
        10000

    Returns
    -------
    {x,y} : pd.DataFrame
        Loaded dataframe, where each column is a unique simulation
    """

    if n_sim > MAX_NSIM:
        raise ValueError(f'Value for n_sim cannot exceed {MAX_NSIM}')
    if sim is None:
        sim = range(n_sim + 1)
    elif np.issubdtype(type(sim), np.integer):
        sim = [0, sim + 1]
    elif hasattr(sim, '__iter__'):
        sim = np.append([0], np.asarray(sim) + 1)
    else:
        raise ValueError('Provided `sim` must be int or array-like')

    # load data for provided `parcellation` and `scale`
    ddir = utils.pathify(alphadir) / parcellation
    x = pd.read_csv(ddir / f'{scale}_x.csv', index_col=0, usecols=sim)
    y = pd.read_csv(ddir / f'{scale}_y.csv', index_col=0, usecols=sim)

    # drop the corpus callosum / unknown / medial wall parcels, if present
    x, y = utils.drop_unknown(x), utils.drop_unknown(y)

    return np.squeeze(x), np.squeeze(y)


def load_vertex_data(alphadir, sim=None, n_sim=MAX_NSIM):
    """
    Loads dense data for specified `alphadir`

    Parameters
    ----------
    alphadir : os.PathLike
        Filepath to directory for desired spatial autocorrelation nulls
    sim : {int, None}, optional
        Which simulation to load. If not specified, will load first `n_sim`
        simulations available. Default: None
    n_sim : int, optional
        Number of simulations to load, if `sim` is not specified. Default:
        10000

    Returns
    -------
    {x,y} : (N, P) np.ndarray
        Data arrays, where each column is a unique simulation
    """

    if n_sim > MAX_NSIM:
        raise ValueError(f'Value for n_sim cannot exceed {MAX_NSIM}')

    if sim is None:
        sims = range(n_sim)
    elif np.issubdtype(type(sim), np.integer):
        sims = range(sim, sim + 1)
    elif hasattr(sim, '__iter__'):
        sims = sim
    else:
        raise ValueError('Provided `sim` must be int or array-like')

    # load data for provided `parcellation` and `scale`
    ddir = utils.pathify(alphadir) / 'sim'
    x, y = np.zeros((20484, len(sims))), np.zeros((20484, len(sims)))
    for n, sim in enumerate(sims):
        x[:, n] = nib.load(ddir / f'x_{sim:04d}.mgh').get_fdata().squeeze()
        y[:, n] = nib.load(ddir / f'y_{sim:04d}.mgh').get_fdata().squeeze()

    # drop the corpus callosum / unknown / medial wall parcels, if present
    mask = np.logical_or(np.all(x == 0, axis=-1), np.all(y == 0, axis=-1))
    x[mask], y[mask] = np.nan, np.nan

    return np.squeeze(x), np.squeeze(y)


def calc_moran(dist, nulls, n_jobs=1):
    """
    Calculates Moran's I for every column of `nulls`

    Parameters
    ----------
    dist : (N, N) array_like
        Full distance matrix (inter-hemispheric distance should be np.inf)
    nulls : (N, P) array_like
        Null brain maps for which to compute Moran's I
    n_jobs : int, optional
        Number of parallel workers to use for calculating Moran's I. Default: 1

    Returns
    -------
    moran : (P,) np.ndarray
        Moran's I for `P` null maps
    """

    def _moran(dist, sim, medmask):
        mask = np.logical_and(medmask, np.logical_not(np.isnan(sim)))
        return spatial.morans_i(dist[np.ix_(mask, mask)], sim[mask],
                                normalize=False, invert_dist=False)

    # do some pre-calculation on our distance matrix to reduce computation time
    with np.errstate(divide='ignore', invalid='ignore'):
        dist = 1 / dist
        np.fill_diagonal(dist, 0)
        dist /= dist.sum(axis=-1, keepdims=True)
    # NaNs in the `dist` array are the "original" medial wall; mask these
    medmask = np.logical_not(np.isnan(dist[:, 0]))

    # calculate moran's I, masking out NaN values for each null (i.e., the
    # rotated medial wall)
    fn = dump(dist, spatial.make_tmpname('.mmap'))[0]
    dist = load(fn, mmap_mode='r')
    moran = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_moran)(dist, nulls[:, n], medmask)
            for n in utils.trange(nulls.shape[-1], desc="Running Moran's I")
        )
    )

    Path(fn).unlink()
    return moran


def load_full_distmat(data, distdir, parcellation, scale):
    """
    Returns full distance matrix for given `parcellation` and `scale`

    Parameters
    ----------
    data : pd.DataFrame or array_like
        Data used to determine hemisphere designations for loaded distance
        matrices
    distdir : os.PathLike
        Filepath to directory containing geodesic distance files
    parcellation : {'atl-cammoun2012', 'atl-schaefer2018'}
        Name of parcellation to use
    scale : str
        Scale of parcellation to use. Must be valid scale for specified
        `parcellation`

    Returns
    -------
    dist : (N, N) np.ndarray
        Full distance matrix (inter-hemispheric distances set to np.inf)
    """

    # get "full" distance matrix for data, with inter-hemi set to np.inf
    dist = np.ones((len(data), len(data))) * np.inf
    for _, hdist, hidx in utils.yield_data_dist(distdir, parcellation,
                                                scale, data, inverse=False):
        dist[np.ix_(hidx, hidx)] = hdist
    np.fill_diagonal(dist, 1)

    return dist
