# -*- coding: utf-8 -*-
"""
Script for running nulls on simulations IN PARALLEL (n.b., there is NO
parallelization at the level of each null method)
"""

from argparse import ArgumentParser
from pathlib import Path
import time

from joblib import Parallel, delayed
import numpy as np

import threadpoolctl

from brainsmash import mapgen
from brainspace.null_models import moran
from netneurotools import (datasets as nndata,
                           freesurfer as nnsurf,
                           stats as nnstats)
from parspin import burt, simnulls, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SPDIR = Path('./data/derivatives/spins').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()

ALPHA = 0.05  # p-value threshold
N_PROC = 24  # number of parallel workers for surrogate generation
N_PERM = 1000  # number of permutations for null models
N_SIM = 1000  # number of simulations to run
SEED = 1234  # reproducibility
SHUFFLE = True  # if we're shuffling sims instead of running paired (r = 0.15)
USE_KNN = False  # whether to use default nearest neigh setting for Burt-2020


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
    for hdata, dist, idx in putils.yield_data_dist(
        DISTDIR, parcellation, scale, data, inverse=(spatnull == 'moran')
    ):

        # handle NaNs before generating surrogates; should only be relevant
        # when using vertex-level data, but good nonetheless
        mask = np.logical_not(np.isnan(hdata))
        surrogates[idx[np.logical_not(mask)]] = np.nan
        hdata, dist, idx = hdata[mask], dist[np.ix_(mask, mask)], idx[mask]

        if spatnull == 'burt2018':
            # Box-Cox transformation requires positive data :man_facepalming:
            hdata += np.abs(dmin) + 0.1
            surrogates[idx] = \
                burt.batch_surrogates(dist, hdata, n_surr=N_PERM, seed=SEED)
        elif spatnull == 'burt2020':
            if parcellation == 'vertex':  # memmap is required for this shit
                index = np.argsort(dist, axis=-1)
                dist = np.sort(dist, axis=-1)
                knn = 1000 if USE_KNN else len(hdata)
                surrogates[idx] = \
                    mapgen.Sampled(hdata, dist, index, knn=knn,
                                   seed=SEED)(N_PERM).T
            else:
                surrogates[idx] = \
                    mapgen.Base(hdata, dist, seed=SEED)(N_PERM, 50).T
        elif spatnull == 'moran':
            mrs = moran.MoranRandomization(joint=True, n_rep=N_PERM,
                                           tol=1e-6, random_state=SEED)
            with threadpoolctl.threadpool_limits(limits=2):
                surrogates[idx] = mrs.fit(dist).randomize(hdata).T

    return surrogates


def _cornblath(x, y, spins, annot):
    """ Calculates p-value using Cornblath method for provided data
    """
    nulls = nnsurf.spin_data(y, version='fsaverage5',
                             lhannot=annot.lh, rhannot=annot.rh,
                             spins=spins, n_rotate=spins.shape[-1])

    return simnulls.calc_pval(x, y, nulls)


def _baum(x, y, spins):
    """ Calculates p-value using Cornblath method for provided data
    """
    nulls = y[spins]
    nulls[spins == -1] = np.nan

    return simnulls.calc_pval(x, y, nulls)


def _genmod(x, y, parcellation, scale, spatnull):
    """ Calculates p-value w/Burt2018/Burt2020/MRS method for provided data
    """
    nulls = make_surrogates(y, parcellation, scale, spatnull)
    return simnulls.calc_pval(x, y, nulls)


def _get_ysim(y, sim):
    """ Gets `sim` column from `y`, accounting for DataFrame vs ndarray
    """
    try:
        return y.iloc[:, sim]
    except AttributeError:
        return y[:, sim]


def run_null(parcellation, scale, spatnull, alpha):
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
    """

    print(f'{time.ctime()}: {parcellation} {scale} {spatnull} {alpha} ',
          flush=True)

    # filenames (for I/O)
    spins_fn = SPDIR / parcellation / spatnull / f'{scale}_spins.csv'
    pvals_fn = (SIMDIR / alpha / parcellation / 'nulls' / spatnull
                / f'{scale}_nulls.csv')
    perms_fn = pvals_fn.parent / f'{scale}_perms.csv'

    if SHUFFLE:
        pvals_fn = pvals_fn.parent / f'{scale}_nulls_shuffle.csv'
        perms_fn = perms_fn.parent / f'{scale}_perms_shuffle.csv'

    if pvals_fn.exists() and perms_fn.exists():
        return

    # load simulated data
    alphadir = SIMDIR / alpha
    if parcellation == 'vertex':
        x, y = simnulls.load_vertex_data(alphadir, n_sim=N_SIM)
    else:
        x, y = simnulls.load_parc_data(alphadir, parcellation, scale,
                                       n_sim=N_SIM)

    # if we're computing info on SHUFFLED data, get the appropriate random `y`
    if SHUFFLE:
        y = _get_ysim(y, np.random.default_rng(1).permutation(N_SIM))

    # calculate the null p-values
    if spatnull == 'naive-para':
        pvals = nnstats.efficient_pearsonr(x, y, nan_policy='omit')[1]
        perms = np.array([np.nan])
    elif spatnull == 'cornblath':
        fn = SPDIR / 'vertex' / 'vazquez-rodriguez' / 'fsaverage5_spins.csv'
        x, y = np.asarray(x), np.asarray(y)
        spins = simnulls.load_spins(fn, n_perm=N_PERM)
        fetcher = getattr(nndata, f"fetch_{parcellation.replace('atl-', '')}")
        annot = fetcher('fsaverage5', data_dir=ROIDIR)[scale]
        out = Parallel(n_jobs=N_PROC, max_nbytes=None)(
            delayed(_cornblath)(x[:, sim], y[:, sim], spins, annot)
            for sim in putils.trange(x.shape[-1], desc='Running simulations')
        )
        pvals, perms = zip(*out)
    elif spatnull == 'baum':
        x, y = np.asarray(x), np.asarray(y)
        spins = simnulls.load_spins(spins_fn, n_perm=N_PERM)
        out = Parallel(n_jobs=N_PROC, max_nbytes=None)(
            delayed(_baum)(x[:, sim], y[:, sim], spins)
            for sim in putils.trange(x.shape[-1], desc='Running simulations')
        )
        pvals, perms = zip(*out)
    elif spatnull in ('burt2018', 'burt2020', 'moran'):
        xarr = np.asarray(x)
        out = Parallel(n_jobs=N_PROC, max_nbytes=None)(
            delayed(_genmod)(xarr[:, sim], _get_ysim(y, sim),
                             parcellation, scale, spatnull)
            for sim in putils.trange(x.shape[-1], desc='Running simulations')
        )
        pvals, perms = zip(*out)
    else:  # vazquez-rodriguez, vasa, hungarian, naive-nonparametric
        x, y = np.asarray(x), np.asarray(y)
        spins = simnulls.load_spins(spins_fn, n_perm=N_PERM)
        out = Parallel(n_jobs=N_PROC, max_nbytes=None)(
            delayed(simnulls.calc_pval)(x[:, sim], y[:, sim], y[spins, sim])
            for sim in putils.trange(x.shape[-1], desc='Running simulations')
        )
        pvals, perms = zip(*out)

    # save to disk
    putils.save_dir(perms_fn, np.atleast_1d(perms), overwrite=False)
    putils.save_dir(pvals_fn, np.atleast_1d(pvals), overwrite=False)


def main():
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    # get inputs
    args = get_parser()

    # reset some stuff
    for param in ('n_perm', 'n_proc', 'seed', 'n_sim', 'shuffle', 'use_knn'):
        globals()[param.upper()] = args[param]

    print(f'N_PERM: {N_PERM}',
          f'N_PROC: {N_PROC}',
          f'SEED: {SEED}',
          f'SHUFFLE: {SHUFFLE}',
          f'N_SIM: {N_SIM}',
          f'SPATNULLS: {args["spatnull"]}',
          f'ALPHAS: {args["alpha"]}\n', sep='\n')

    if args['show_params']:
        return

    # everyone loves a four-level-deep nested for-loop :man_facepalming:
    for spatnull in args['spatnull']:
        for alpha in args['alpha']:
            if spatnull in simnulls.VERTEXWISE:
                run_null('vertex', 'fsaverage5', spatnull, alpha)
            for parcellation, annotations in parcellations.items():
                for scale in annotations:
                    run_null(parcellation, scale, spatnull, alpha)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--show_params', default=False, action='store_true')
    parser.add_argument('--n_perm', default=N_PERM, type=int)
    parser.add_argument('--n_proc', default=N_PROC, type=int)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--use_knn', default=False, action='store_true')
    parser.add_argument('--spatnull', choices=simnulls.SPATNULLS,
                        default=simnulls.SPATNULLS, nargs='+')
    parser.add_argument('--alpha', choices=simnulls.ALPHAS,
                        default=simnulls.ALPHAS, nargs='+')
    parser.add_argument('n_sim', type=int, default=N_SIM, nargs='?')
    return vars(parser.parse_args())


if __name__ == "__main__":
    main()
