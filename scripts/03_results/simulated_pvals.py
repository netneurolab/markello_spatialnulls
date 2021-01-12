# -*- coding: utf-8 -*-
"""
Script for running nulls on simulations SERIALLY (n.b., parallelization may be
done at the level of each null method, but simulations are run individually)
"""

from collections import defaultdict
from dataclasses import asdict, make_dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from netneurotools import stats as nnstats
from parspin import simnulls, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()

SEED = 1234  # reproducibility
SIM = 9999  # which simulation was used to generate 10000 nulls
N_PVALS = 1000  # how many repeated draws should be done to calculate pvals

NullResult = make_dataclass(
    'NullResult', ('parcellation', 'scale', 'spatnull', 'alpha', 'prob')
)


def pval_from_perms(actual, null):
    """ Calculates p-value of `actual` based on `null` permutations
    """

    return (np.sum(np.abs(null) >= np.abs(actual)) + 1) / (len(null) + 1)


def pval_by_subsets(parcellation, scale, spatnull, alpha):
    """
    Parameters
    ----------
    parcellation : {'vertex', 'atl-cammoun2012', 'atl-schaefer2018'}
    scale : str
    spatnull : str
    alpha : str

    Returns
    -------
    pvals : pd.DataFrame
    """

    # load simulated data
    alphadir = SIMDIR / alpha
    if parcellation == 'vertex':
        x, y = simnulls.load_vertex_data(alphadir, sim=SIM)
    else:
        x, y = simnulls.load_parc_data(alphadir, parcellation, scale, sim=SIM)

    corr = nnstats.efficient_pearsonr(x, y, nan_policy='omit')[0]
    perms = np.loadtxt(alphadir / parcellation / 'nulls' / spatnull / 'pvals'
                       / f'{scale}_perms_{SIM}.csv')

    pvals = defaultdict(list)
    for subset in [100, 500, 1000, 5000]:
        rs = np.random.default_rng(SEED)
        for n in range(N_PVALS):
            # select `subset` correlations from `perms` and calculate p-value
            # store the p-value and repeat `N_PVALS` times
            sub = rs.choice(perms, size=subset, replace=False)
            pvals[subset].append(pval_from_perms(corr, sub))
        # arrays are nicer than lists
        pvals[subset] = np.asarray(pvals[subset])

    df = pd.melt(pd.DataFrame(pvals), var_name='n_nulls', value_name='p_value')
    # add single p-value generated from 10000 nulls
    df = df.append({
        'n_nulls': len(perms), 'p_value': pval_from_perms(corr, perms)},
        ignore_index=True
    ).assign(
        parcellation=parcellation, scale=scale, spatnull=spatnull, alpha=alpha
    )

    return df


def get_prob(parcellation, scale, spatnull, alpha):
    """ Gets probability of p-values of simulations for given null being < 0.05
    """

    nulldir = SIMDIR / alpha / parcellation / 'nulls' / spatnull
    pvals_fn = nulldir / f'{scale}_nulls.csv'
    prob = np.sum(np.loadtxt(pvals_fn) < 0.05)

    return asdict(NullResult(parcellation, scale, spatnull, alpha, prob))


def main():
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    pvals = pd.DataFrame(columns=['parcellation', 'scale', 'spatnull', 'alpha',
                                  'n_nulls', 'p_value'])
    nullresults = []
    for spatnull in simnulls.SPATNULLS:
        if spatnull == 'naive-para':  # can't do that one :man_shrugging:
            continue
        for alpha in simnulls.ALPHAS:
            if spatnull in simnulls.VERTEXWISE:
                pvals = pvals.append(
                    pval_by_subsets('vertex', 'fsaverage5', spatnull, alpha),
                    ignore_index=True, sort=True
                )
                nullresults.append(
                    get_prob('vertex', 'fsaverage5', spatnull, alpha)
                )
            for parcellation, annotations in parcellations.items():
                for scale in annotations:
                    pvals = pvals.append(
                        pval_by_subsets(parcellation, scale, spatnull, alpha),
                        ignore_index=True, sort=True
                    )
                    nullresults.append(
                        get_prob('vertex', 'fsaverage5', spatnull, alpha)
                    )
