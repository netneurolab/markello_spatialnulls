# -*- coding: utf-8 -*-
"""
Script for running null models on parcellated simulated data
"""

from dataclasses import asdict, make_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from brainsmash import mapgen
from brainspace.null_models import moran
from netneurotools import (datasets as nndata,
                           freesurfer as nnsurf,
                           stats as nnstats)
from parspin import burt, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SPDIR = Path('./data/derivatives/spins').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()
SPATNULLS = [
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
ALPHA = 0.05
ALPHAS = np.arange(0, 3.5, 0.5)
N_PROC = 36
N_PERM = 10000
SEED = 1234

NullResult = make_dataclass('null', (
    'parcellation', 'scale', 'spatnull', 'alpha', 'prob'
))


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

    x, y, nulls = np.asarray(x), np.asarray(y), np.asarray(nulls)

    # calculate real + permuted correlation coefficients
    real = np.corrcoef(x, y)[0, 1]
    perms = nnstats.efficient_pearsonr(x, nulls)[0]
    pval = (np.sum(np.abs(perms) >= np.abs(real)) + 1) / (len(perms) + 1)

    return pval


def load_data(parcellation, scale, alpha):
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


def make_surrogates(data, parcellation, scale, spatnull):
    """
    Generates surrogates for `data` using `spatnull` method

    Parameters
    ----------
    data : (N,) pd.DataFrame
    parcellation : {'atl-cammoun2012', 'atl-schaefer2018''}
    scale : str
    spatnull : {'burt2018', 'burt2020', 'moran'}

    Returns
    -------
    surrogates : (N, `N_PERM`) np.ndarray
    """

    if spatnull not in ('burt2018', 'burt2020', 'moran'):
        raise ValueError(f'Cannot make surrogates for null method {spatnull}')

    surrogates = np.zeros((len(data), N_PERM))
    for hdata, dist, idx in putils.yield_data_dist(DISTDIR, parcellation,
                                                   scale, data, inverse=False):
        if spatnull == 'burt2018':
            surrogates[idx] = \
                burt.batch_surrogates(dist, hdata, n_surr=N_PERM,
                                      n_jobs=N_PROC, seed=SEED)
        elif spatnull == 'burt2020':
            surrogates[idx] = \
                mapgen.Base(hdata, dist, seed=SEED, n_jobs=N_PROC)(N_PERM, 200)
        elif spatnull == 'moran':
            dist = (dist + np.eye(len(dist))) ** -1
            mrs = moran.MoranRandomization(joint=True, n_rep=N_PERM,
                                           tol=1e-6, random_state=SEED)
            surrogates[idx] = mrs.fit(dist).randomize(hdata)

    return surrogates


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

    Returns
    -------
    stats : dict
        With keys 'parcellation', 'scale', 'spatnull', 'alpha', and 'pval'
    """

    x, y = load_data(parcellation, scale, alpha)
    out = (SIMDIR / alpha / parcellation / 'nulls' / spatnull
           / f'{scale}_nulls.csv')

    pvals = np.zeros(x.shape[-1])
    if out.exists():
        pvals = np.loadtxt(out).reshape(-1, 1)
    elif spatnull == 'naive-para':
        pvals = nnstats.efficient_pearsonr(x, y)[1]
    elif spatnull == 'cornblath':
        fn = SPDIR / 'vertex' / 'vazquez-rodriguez' / 'fsaverage5_spins.csv'
        spins = np.loadtxt(fn, delimiter=',', dtype='int32')
        fetcher = getattr(nndata, f"fetch_{parcellation.replace('atl-', '')}")
        annotations = fetcher('fsaverage5', data_dir=ROIDIR)[scale]
        for sim in range(x.shape[-1]):
            nulls = nnsurf.spin_data(y[:, sim], version='fsaverage5',
                                     lhannot=annotations.lh,
                                     rhannot=annotations.rh,
                                     spin=spins, n_rotate=spins.shape[-1])
            pvals[sim] = calc_pval(x[:, sim], y[:, sim], nulls)
    elif spatnull == 'baum':
        fn = SPDIR / parcellation / spatnull / f'{scale}_spins.csv'
        spins = np.loadtxt(fn, delimiter=',', dtype='int32')
        x, y = np.asarray(x), np.asarray(y)
        perms = np.zeros((x.shape[-1], spins.shape[-1]))
        for n, spin in enumerate(spins.T):
            mask = spin != -1
            perms[:, n] = nnstats.efficient_pearsonr(x[mask], y[spin[mask]])[0]
        real = np.abs(nnstats.efficient_pearsonr(x, y)[0][:, None])
        pvals = ((np.sum(np.abs(perms) >= real, axis=1) + 1)
                 / (perms.shape[-1] + 1))
    elif spatnull in ('burt2018', 'burt2020', 'moran'):
        for sim in range(x.shape[-1]):
            nulls = make_surrogates(y[:, sim], parcellation, scale, spatnull)
            pvals[sim] = calc_pval(x[:, sim], y[:, sim], nulls)
    else:  # vazquez-rodriguez, vasa, hungarian, naive-nonparametric
        fn = SPDIR / parcellation / spatnull / f'{scale}_spins.csv'
        spins = np.loadtxt(fn, delimiter=',', dtype='int32')
        pvals = np.asarray([
            calc_pval(x[:, n], y[:, n], y[spins, n]) for n in range(len(x.T))
        ])

    # checkpoint our hard work!
    putils.save_dir(out, pvals, overwrite=False)

    # calculate probability that p < 0.05
    prob = np.sum(pvals < ALPHA) / len(pvals)
    return asdict(NullResult(parcellation, scale, spatnull, alpha, prob))


def main():
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)
    data = []

    # everyone loves a four-level-deep nested for-loop :man_facepalming:
    for parcellation, annotations in parcellations.items():
        for scale in annotations:
            for spatnull in SPATNULLS:
                for alpha in ALPHAS:
                    alpha = f'{float(alpha):.2f}'  # string-ify
                    data.append(
                        run_null(parcellation, scale, spatnull, alpha),
                    )

    pd.DataFrame(data).to_csv(SIMDIR / 'summary.csv', index=False)


if __name__ == "__main__":
    main()
