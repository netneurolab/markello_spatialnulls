# -*- coding: utf-8 -*-
"""
Script for running nulls on simulations SERIALLY (n.b., parallelization may be
done at the level of each null method, but simulations are run individually)
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from netneurotools import stats as nnstats
from parspin import simnulls, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()
OUTDIR = Path('./data/derivatives/supplementary/comp_nnulls')

SEED = 1234  # reproducibility
SIM = 9999  # which simulation was used to generate 10000 nulls
N_PVALS = 1000  # how many repeated draws should be done to calculate pvals


def pval_from_perms(actual, null):
    """ Calculates p-value of `actual` based on `null` permutations
    """

    return (np.sum(np.abs(null) >= np.abs(actual)) + 1) / (len(null) + 1)


def pval_by_subsets(parcellation, scale, spatnull, alpha):
    """
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
    pvals : pd.DataFrame
    """

    print(spatnull, alpha, parcellation, scale)

    if spatnull == 'naive-para':
        return

    # load simulated data
    alphadir = SIMDIR / alpha
    if parcellation == 'vertex':
        x, y = simnulls.load_vertex_data(alphadir, sim=SIM)
    else:
        x, y = simnulls.load_parc_data(alphadir, parcellation, scale, sim=SIM)

    corr = nnstats.efficient_pearsonr(x, y, nan_policy='omit')[0]
    perms = np.loadtxt(alphadir / parcellation / 'nulls' / spatnull / 'pvals'
                       / f'{scale}_perms_{SIM}.csv')

    orig = pval_from_perms(corr, perms)
    pvals = defaultdict(list)
    for subset in [100, 500, 1000, 5000]:
        rs = np.random.default_rng(SEED)
        for n in range(N_PVALS):
            # select `subset` correlations from `perms` and calculate p-value
            # store the p-value and repeat `N_PVALS` times
            sub = rs.choice(perms, size=subset, replace=False)
            pvals[subset].append(pval_from_perms(corr, sub) - orig)
        # arrays are nicer than lists
        pvals[subset] = np.asarray(pvals[subset])

    df = pd.melt(pd.DataFrame(pvals), var_name='n_nulls', value_name='pval')
    # add single p-value generated from 10000 nulls
    df = df.assign(
        parcellation=parcellation,
        scale=scale,
        spatnull=spatnull,
        alpha=alpha
    )

    order = ['parcellation', 'scale', 'spatnull', 'alpha', 'n_nulls', 'pval']
    return df[order]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    subsets = []
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)
    for spatnull in simnulls.SPATNULLS:
        for alpha in simnulls.ALPHAS:
            if spatnull in simnulls.VERTEXWISE:
                subsets.append(
                    pval_by_subsets('vertex', 'fsaverage5', spatnull, alpha),
                )
            for parcellation, annotations in parcellations.items():
                for scale in annotations:
                    subsets.append(
                        pval_by_subsets(parcellation, scale, spatnull, alpha),
                    )
    subsets = pd.concat(subsets, ignore_index=True, sort=True)
    subsets.to_csv(OUTDIR / 'nnulls_summary.csv', index=False)


if __name__ == "__main__":
    main()
