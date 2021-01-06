# -*- coding: utf-8 -*-
"""
Script for running null models on parcellated simulated data
"""

from argparse import ArgumentParser
from pathlib import Path

from joblib import Parallel, delayed, dump, load
import nibabel as nib
import numpy as np
import pandas as pd
import threadpoolctl

from brainsmash import mapgen
from brainspace.null_models import moran
from netneurotools import (datasets as nndata,
                           freesurfer as nnsurf,
                           stats as nnstats)
from parspin import burt, spatial, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SPDIR = Path('./data/derivatives/spins').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()
SPATNULLS = [  # all our null models we want to run
    'naive-para',
    'naive-nonpara',
    'vazquez-rodriguez',
    'vasa',
    'hungarian',
    'baum',
    'cornblath',
    'burt2018',
    'burt2020',
    'moran'
]
VERTEXWISE = [  # we're only running these ones at the vertex level
    'naive-para',
    'naive-nonpara',
    'vazquez-rodriguez',
    'burt2018',
    'burt2020',
    'moran'
]
ALPHAS = [  # spatial autocorrelation params
    0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0
]
ALPHA = 0.05  # p-value threshold
N_PROC = 12  # number of parallel workers for surrogate generation
N_PERM = 10000  # number of permutations for null models
SEED = 1234  # reproducibility


def load_parc_data(parcellation, scale, alpha):
    """
    Loads data for specified `parcellation`, `scale`, and `alpha`

    Parameters
    ----------
    parcellation : {'atl-cammoun2012', 'atl-schaefer2018'}
        Name of parcellation to use
    scale : str
        Scale of parcellation to use. Must be valid scale for specified `parc`
    alpha : float
        Spatial autocorrelation parameter

    Returns
    -------
    {x,y} : pd.DataFrame
        Loaded dataframe, where each column is a unique simulation
    """

    # load data for provided `parcellation` and `scale`
    ddir = SIMDIR / alpha / parcellation
    x = pd.read_csv(ddir / f'{scale}_x.csv', index_col=0)
    y = pd.read_csv(ddir / f'{scale}_y.csv', index_col=0)

    # drop the corpus callosum / unknown / medial wall parcels, if present
    x, y = putils.drop_unknown(x), putils.drop_unknown(y)

    return x, y


def load_vertex_data(alpha, n_sim=10000):
    """
    Loads dense data for specified `alpha`

    Parameters
    ----------
    alpha : float
        Spatial autocorrelation parameter

    Returns
    -------
    {x,y} : (N, P) np.ndarray
        Data arrays, where each column is a unique simulation
    """

    # load data for provided `parcellation` and `scale`
    ddir = SIMDIR / alpha / 'sim'
    x, y = np.zeros((20484, n_sim)), np.zeros((20484, n_sim))
    for sim in range(N_PERM):
        x[:, sim] = nib.load(ddir / f'x_{sim:04d}.mgh').get_fdata().squeeze()
        y[:, sim] = nib.load(ddir / f'y_{sim:04d}.mgh').get_fdata().squeeze()

    # drop the corpus callosum / unknown / medial wall parcels, if present
    mask = np.logical_or(np.all(x == 0, axis=-1), np.all(y == 0, axis=-1))
    x[mask], y[mask] = np.nan, np.nan

    return x, y


def make_surrogates(data, parcellation, scale, spatnull):
    """
    Generates surrogates for `data` using `spatnull` method

    Parameters
    ----------
    data : (N,) pd.DataFrame
    parcellation : {'atl-cammoun2012', 'atl-schaefer2018'}
    scale : str
    spatnull : {'burt2018', 'burt2020', 'moran'}

    Returns
    -------
    surrogates : (N, `N_PERM`) np.ndarray
    """

    if spatnull not in ('burt2018', 'burt2020', 'moran'):
        raise ValueError(f'Cannot make surrogates for null method {spatnull}')

    darr = np.asarray(data)
    dmin = darr[np.logical_not(np.isnan(darr))].min()

    surrogates = np.zeros((len(data), N_PERM))
    for hdata, dist, idx in putils.yield_data_dist(DISTDIR, parcellation,
                                                   scale, data, inverse=False):

        # handle NaNs before generating surrogates; should only be relevant
        # when using vertex-level data, but good nonetheless
        mask = np.logical_not(np.isnan(hdata))
        surrogates[idx[np.logical_not(mask)]] = np.nan
        hdata, dist, idx = hdata[mask], dist[np.ix_(mask, mask)], idx[mask]

        if spatnull == 'burt2018':
            fn = dump(dist, spatial.make_tmpname('.mmap'))[0]
            dist = load(fn, mmap_mode='r')
            # Box-Cox transformation requires positive data :man_facepalming:
            hdata += np.abs(dmin) + 0.1
            surrogates[idx] = \
                burt.batch_surrogates(dist, hdata, n_surr=N_PERM,
                                      n_jobs=N_PROC, seed=SEED)
        elif spatnull == 'burt2020':
            if parcellation == 'vertex':  # memmap is required for this shit
                fn = dump(dist, spatial.make_tmpname('.mmap'))[0]
                dist = load(fn, mmap_mode='r')
                index = np.argsort(dist, axis=-1)
                dist = np.sort(dist, axis=-1)
                surrogates[idx] = \
                    mapgen.Sampled(hdata, dist, index,
                                   seed=SEED, n_jobs=N_PROC)(N_PERM).T
            else:
                surrogates[idx] = \
                    mapgen.Base(hdata, dist,
                                seed=SEED, n_jobs=N_PROC)(N_PERM, 50).T
        elif spatnull == 'moran':
            np.fill_diagonal(dist, 1)
            dist **= -1
            mrs = moran.MoranRandomization(joint=True, n_rep=N_PERM,
                                           tol=1e-6, random_state=SEED)
            with threadpoolctl.threadpool_limits(limits=N_PROC):
                surrogates[idx] = mrs.fit(dist).randomize(hdata).T

    return surrogates


def load_distmat(data, parcellation, scale):
    """
    Returns full distance matrix for given `parcellation` and `scale`

    Parameters
    ----------
    data : array_like
    parcellation : {'atl-cammoun2012', 'atl-schaefer2018'}
    scale : str

    Returns
    -------
    dist : (N, N) np.ndarray
        Full distance matrix (inter-hemispheric distances set to np.inf)
    """

    # get "full" distance matrix for data, with inter-hemi set to np.inf
    dist = np.ones((len(data), len(data))) * np.inf
    for _, hdist, hidx in putils.yield_data_dist(DISTDIR, parcellation,
                                                 scale, data, inverse=False):
        dist[np.ix_(hidx, hidx)] = hdist
    np.fill_diagonal(dist, 1)

    return dist


def calc_moran(dist, nulls, fname):
    """
    Calculates Moran's I for every column of `nulls`

    Parameters
    ----------
    dist : (N, N) array_like
        Full distance matrix (inter-hemispheric distance should be np.inf)
    nulls : (N, P) array_like
        Null brain maps for which to compute Moran's I

    Returns
    -------
    moran : (P,) np.ndarray
        Moran's I for `P` null maps
    """

    def _moran(dist, sim, medmask):
        mask = np.logical_and(medmask, np.logical_not(np.isnan(sim)))
        return spatial.morans_i(dist[np.ix_(mask, mask)], sim[mask],
                                normalize=False, invert_dist=False)

    if fname.exists():
        return

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
        Parallel(n_jobs=N_PROC, max_nbytes=None)(
            delayed(_moran)(dist, nulls[:, n], medmask)
            for n in putils.trange(nulls.shape[-1],
                                   desc="Calculating Moran's I")
        )
    )

    putils.save_dir(fname, moran, overwrite=False)


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
    """

    x, y, nulls = np.asanyarray(x), np.asanyarray(y), np.asanyarray(nulls)

    # calculate real + permuted correlation coefficients
    real = nnstats.efficient_pearsonr(x, y, nan_policy='omit')[0]
    perms = nnstats.efficient_pearsonr(x, nulls, nan_policy='omit')[0]
    pval = (np.sum(np.abs(perms) >= np.abs(real)) + 1) / (len(perms) + 1)

    return pval


def _load_spins(x, y, fn):
    """ Loads spins from `fn` and return all inputs as arrays
    """
    spins = np.loadtxt(fn, delimiter=',', dtype='int32')
    return np.asarray(x), np.asarray(y), spins


def _get_ysim(y, sim):
    """ Gets `sim` column from `y`, accounting for DataFrame vs ndarray
    """
    try:
        return y.iloc[:, sim]
    except AttributeError:
        return y[:, sim]


def run_null(parcellation, scale, spatnull, alpha, sim):
    """
    Runs spatial null models for given combination of inputs

    Parameters
    ----------
    parcellation : str
        Name of parcellation to be used
    scale : str
        Scale of `parcellation` to be used
    spatnull : str
        Name of spin method to be used
    alpha : float
        Spatial autocorrelation parameter to be used

    Returns
    -------
    stats : dict
        With keys 'parcellation', 'scale', 'spatnull', 'alpha', and 'prob',
        where 'prob' is the probability that the p-value for a given simulation
        is less than ALPHA (across all simulations)
    """

    print(f'JOB: {parcellation} {scale} {spatnull} {alpha} {sim}', flush=True)

    # filenames (for I/O)
    spins_fn = SPDIR / parcellation / spatnull / f'{scale}_spins.csv'
    pvals_fn = (SIMDIR / alpha / parcellation / 'nulls' / spatnull
                / 'pvals' / f'{scale}_nulls_{sim:04d}.csv')
    moran_fn = pvals_fn.parent / f'{scale}_moran.csv'

    # load simulated data
    if parcellation == 'vertex':
        x, y = load_vertex_data(alpha)
    else:
        x, y = load_parc_data(parcellation, scale, alpha)

    if sim == -1:
        dist = load_distmat(y, parcellation, scale)

    # calculate the null p-values
    if pvals_fn.exists():
        pvals = np.loadtxt(pvals_fn)
    elif spatnull == 'naive-para':
        pvals = nnstats.efficient_pearsonr(x, y, nan_policy='omit')[1]
        return
    elif spatnull == 'cornblath':
        fn = SPDIR / 'vertex' / 'vazquez-rodriguez' / 'fsaverage5_spins.csv'
        x, y, spins = _load_spins(x, y, fn)
        fetcher = getattr(nndata, f"fetch_{parcellation.replace('atl-', '')}")
        annot = fetcher('fsaverage5', data_dir=ROIDIR)[scale]
        nulls = nnsurf.spin_data(y[:, sim], version='fsaverage5',
                                 lhannot=annot.lh, rhannot=annot.rh,
                                 spins=spins, n_rotate=spins.shape[-1])
        xsim, ysim = x[:, sim], y[:, sim]
    elif spatnull == 'baum':
        x, y, spins = _load_spins(x, y, spins_fn)
        nulls = y[spins, sim]
        nulls[spins == -1] = np.nan
        xsim, ysim = x[:, sim], y[:, sim]
    elif spatnull in ('burt2018', 'burt2020', 'moran'):
        # we can't parallelize this because `make_surrogates()` is parallelized
        xsim = np.asarray(x)[:, sim]
        ysim = _get_ysim(y, sim)
        nulls = make_surrogates(ysim, parcellation, scale, spatnull)
    else:  # vazquez-rodriguez, vasa, hungarian, naive-nonparametric
        x, y, spins = _load_spins(x, y, spins_fn)
        nulls = y[spins, sim]
        xsim, ysim = x[:, sim], y[:, sim]

    pvals = calc_pval(xsim, ysim, nulls)
    putils.save_dir(pvals_fn, np.atleast_1d(pvals), overwrite=False)

    if sim == -1:
        calc_moran(dist, nulls, moran_fn)


def main():
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    # this chunk of code is only relevant if you're trying to run on an HPC
    args = _get_parser()

    alpha = f"alpha-{float(args['alpha']):.1f}"
    if args['start'] is not None:
        sims = range(args['start'], args['start'] + args['sim'])
    else:
        sims = range(args['sim'], args['sim'] + 1)

    for sim in sims:
        if args['spatnull'] in VERTEXWISE:
            run_null('vertex', 'fsaverage5', args['spatnull'], alpha, sim)
        for parcellation, annotations in parcellations.items():
            for scale in annotations:
                run_null(parcellation, scale, args['spatnull'], alpha, sim)


def _get_parser():
    parser = ArgumentParser()
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('spatnull', help='Spatial null method')
    parser.add_argument('alpha', help='Spatial autocorrelation parameter')
    parser.add_argument('sim', type=int, help='Which simulation to run')
    return vars(parser.parse_args())


if __name__ == "__main__":
    main()
