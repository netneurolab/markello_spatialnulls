# -*- coding: utf-8 -*-
"""
Script for running nulls on simulations SERIALLY (n.b., parallelization may be
done at the level of each null method, but simulations are run individually)
"""

from argparse import ArgumentParser
from pathlib import Path
import time

from joblib import dump, load
import numpy as np
import threadpoolctl

from brainsmash import mapgen
from brainspace.null_models import moran
from netneurotools import (datasets as nndata,
                           freesurfer as nnsurf,
                           stats as nnstats)
from parspin import burt, simnulls, spatial, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SPDIR = Path('./data/derivatives/spins').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()

ALPHA = 0.05  # p-value threshold
N_PROC = 12  # number of parallel workers for surrogate generation
N_PERM = 1000  # number of permutations for null models
SEED = 1234  # reproducibility
RUN_MORAN = False  # calculate Moran's I?


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
            fn = dump(dist, spatial.make_tmpname('.mmap'))[0]
            dist = load(fn, mmap_mode='r')
            # Box-Cox transformation requires positive data :man_facepalming:
            hdata += np.abs(dmin) + 0.1
            surrogates[idx] = \
                burt.batch_surrogates(dist, hdata, n_surr=N_PERM,
                                      n_jobs=N_PROC, seed=SEED)
            Path(fn).unlink()
        elif spatnull == 'burt2020':
            if parcellation == 'vertex':  # memmap is required for this shit
                fn = dump(dist, spatial.make_tmpname('.mmap'))[0]
                dist = load(fn, mmap_mode='r')
                index = np.argsort(dist, axis=-1)
                dist = np.sort(dist, axis=-1)
                surrogates[idx] = \
                    mapgen.Sampled(hdata, dist, index,
                                   seed=SEED, n_jobs=N_PROC)(N_PERM).T
                Path(fn).unlink()
            else:
                surrogates[idx] = \
                    mapgen.Base(hdata, dist,
                                seed=SEED, n_jobs=N_PROC)(N_PERM, 50).T
        elif spatnull == 'moran':
            mrs = moran.MoranRandomization(joint=True, n_rep=N_PERM,
                                           tol=1e-6, random_state=SEED)
            with threadpoolctl.threadpool_limits(limits=N_PROC):
                surrogates[idx] = mrs.fit(dist).randomize(hdata).T

    return surrogates


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
    sim : int
        Which simulation to run

    Returns
    -------
    stats : dict
        With keys 'parcellation', 'scale', 'spatnull', 'alpha', and 'prob',
        where 'prob' is the probability that the p-value for a given simulation
        is less than ALPHA (across all simulations)
    """

    print(f'{time.ctime()}: {parcellation} {scale} {spatnull} {alpha} '
          f'sim-{sim} ', flush=True)

    # filenames (for I/O)
    spins_fn = SPDIR / parcellation / spatnull / f'{scale}_spins.csv'
    pvals_fn = (SIMDIR / alpha / parcellation / 'nulls' / spatnull
                / 'pvals' / f'{scale}_nulls_{sim:04d}.csv')
    perms_fn = pvals_fn.parent / f'{scale}_perms_{sim:04d}.csv'
    moran_fn = pvals_fn.parent / f'{scale}_moran_{sim:04d}.csv'

    # load simulated data
    alphadir = SIMDIR / alpha
    if parcellation == 'vertex':
        x, y = simnulls.load_vertex_data(alphadir, sim=sim)
    else:
        x, y = simnulls.load_parc_data(alphadir, parcellation, scale, sim=sim)

    # if we're going to run moran for this simulation, pre-load distmat
    if RUN_MORAN and not moran_fn.exists():
        dist = simnulls.load_full_distmat(y, DISTDIR, parcellation, scale)

    # calculate the null p-values
    nulls = None
    if pvals_fn.exists() and perms_fn.exists():
        pvals, perms = np.loadtxt(pvals_fn), np.loadtxt(perms_fn)
    elif spatnull == 'naive-para':
        pvals = nnstats.efficient_pearsonr(x, y, nan_policy='omit')[1]
        perms = np.array([np.nan])
    elif spatnull == 'cornblath':
        fn = SPDIR / 'vertex' / 'vazquez-rodriguez' / 'fsaverage5_spins.csv'
        x, y = np.asarray(x), np.asarray(y)
        spins = simnulls.load_spins(fn, n_perm=N_PERM)
        fetcher = getattr(nndata, f"fetch_{parcellation.replace('atl-', '')}")
        annot = fetcher('fsaverage5', data_dir=ROIDIR)[scale]
        nulls = nnsurf.spin_data(y, version='fsaverage5',
                                 lhannot=annot.lh, rhannot=annot.rh,
                                 spins=spins, n_rotate=spins.shape[-1])
        pvals, perms = simnulls.calc_pval(x, y, nulls)
    elif spatnull == 'baum':
        x, y = np.asarray(x), np.asarray(y)
        spins = simnulls.load_spins(spins_fn, n_perm=N_PERM)
        nulls = y[spins]
        nulls[spins == -1] = np.nan
        pvals, perms = simnulls.calc_pval(x, y, nulls)
    elif spatnull in ('burt2018', 'burt2020', 'moran'):
        nulls = make_surrogates(y, parcellation, scale, spatnull)
        pvals, perms = simnulls.calc_pval(x, y, nulls)
    else:  # vazquez-rodriguez, vasa, hungarian, naive-nonparametric
        x, y = np.asarray(x), np.asarray(y)
        spins = simnulls.load_spins(spins_fn, n_perm=N_PERM)
        nulls = y[spins]
        pvals, perms = simnulls.calc_pval(x, y, nulls)

    # save to disk
    putils.save_dir(perms_fn, np.atleast_1d(perms), overwrite=False)
    putils.save_dir(pvals_fn, np.atleast_1d(pvals), overwrite=False)

    # if we're running moran, do it now
    if RUN_MORAN and not moran_fn.exists() and nulls is not None:
        moran = simnulls.calc_moran(dist, nulls, n_jobs=N_PROC)
        putils.save_dir(moran_fn, np.atleast_1d(moran), overwrite=False)


def main():
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    # get inputs
    args = get_parser()

    # reset some stuff
    config = globals()
    config['N_PERM'] = args['n_perm']
    config['N_PROC'] = args['n_proc']
    config['SEED'] = args['seed']
    config['RUN_MORAN'] = args['run_moran']

    sims = range(args['start'], args['start'] + args['n_sim'])

    print(f'N_PERM: {N_PERM}\tN_PROC: {N_PROC}\tSEED: {SEED}\t'
          f'START: {sims.start}\tSTOP:{sims.stop}')

    for spatnull in args['spatnull']:
        for alpha in args['alpha']:
            # no parallelization here
            for sim in sims:
                # maybe parallelization here
                if spatnull in simnulls.VERTEXWISE:
                    run_null('vertex', 'fsaverage5', spatnull, alpha, sim)
                for parcellation, annotations in parcellations.items():
                    for scale in annotations:
                        run_null(parcellation, scale, spatnull, alpha, sim)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--n_perm', default=N_PERM, type=int)
    parser.add_argument('--n_proc', default=N_PROC, type=int)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--run_moran', default=False, action='store_true')
    parser.add_argument('--spatnull', choices=simnulls.SPATNULLS,
                        default=simnulls.SPATNULLS, nargs='+')
    parser.add_argument('--alpha', choices=simnulls.ALPHAS,
                        default=simnulls.ALPHAS, nargs='+')
    parser.add_argument('start', type=int, default=0, nargs='?')
    parser.add_argument('n_sim', type=int, default=1, nargs='?')
    return vars(parser.parse_args())


if __name__ == "__main__":
    main()
