# -*- coding: utf-8 -*-
"""
Script to combine outputs of `run_simulated_nulls_serial.py` once all nulls +
simulations have completed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from netneurotools import stats as nnstats
from parspin import simnulls, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()

N_PERM = 1000  # number of permutations for null models
N_SIM = 1000  # number of simulations to run


def combine_nulls(parcellation, scale, spatnull, alpha):
    """
    Combines outputs of all simulations into single files for provided inputs

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

    print(f'{spatnull} {alpha} {parcellation} {scale}')

    nulldir = SIMDIR / alpha / parcellation / 'nulls' / spatnull
    pvals_fn = nulldir / f'{scale}_nulls.csv'
    perms_fn = nulldir / f'{scale}_perms.csv'

    # only some of the spatial null models were run in serial mode; these are
    # the ones that are missing the top-level file and whose outputs we need to
    # combine. do that here.
    if not pvals_fn.exists():
        pvals, perms = np.zeros(N_SIM), np.zeros((N_PERM, N_SIM))
        for sim in range(N_SIM):
            pvals[sim] = \
                np.loadtxt(nulldir / 'pvals' / f'{scale}_nulls_{sim:04d}.csv')
            perms[:, sim] = \
                np.loadtxt(nulldir / 'pvals' / f'{scale}_perms_{sim:04d}.csv')
        putils.save_dir(pvals_fn, pvals, overwrite=False)
        putils.save_dir(perms_fn, perms, overwrite=False)
    else:
        pvals = np.loadtxt(pvals_fn)

    # grab the empirical correlations for each simulation---good to have
    if parcellation == 'vertex':
        x, y = simnulls.load_vertex_data(SIMDIR / alpha, n_sim=N_SIM)
    else:
        x, y = simnulls.load_parc_data(SIMDIR / alpha, parcellation, scale,
                                       n_sim=N_SIM)
    corrs = nnstats.efficient_pearsonr(x, y, nan_policy='omit')[0]

    return pd.DataFrame(dict(
        parcellation=parcellation,
        scale=scale,
        spatnull=spatnull,
        alpha=alpha,
        corr=corrs,
        sim=range(len(pvals)),
        pval=pvals
    ))


def combine_shuffle(parcellation, scale, spatnull, alpha):
    """
    Combines outputs of all simulations into single files for provided inputs

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

    nulldir = SIMDIR / alpha / parcellation / 'nulls' / spatnull
    pvals_fn = nulldir / f'{scale}_nulls_shuffle.csv'
    perms_fn = nulldir / f'{scale}_perms_shuffle.csv'

    # only some of the spatial null models were run in serial mode; these are
    # the ones that are missing the top-level file and whose outputs we need to
    # combine. do that here.
    if not pvals_fn.exists():
        pvals, perms = np.zeros(N_SIM), np.zeros((N_PERM, N_SIM))
        for sim in range(N_SIM):
            pvals[sim] = np.loadtxt(
                nulldir / 'pvals' / f'{scale}_nulls_shuffle_{sim:04d}.csv'
            )
            perms[:, sim] = np.loadtxt(
                nulldir / 'pvals' / f'{scale}_perms_shuffle_{sim:04d}.csv'
            )
        putils.save_dir(pvals_fn, pvals, overwrite=False)
        putils.save_dir(perms_fn, perms, overwrite=False)
    else:
        pvals = np.loadtxt(pvals_fn)

    # the "shuffled" p-values were generated so we could calculate the
    # Prob(p < 0.05) of each null / alpha combination (i.e., the FWER)
    prob = np.sum(pvals < 0.05) / len(pvals)

    return dict(
        parcellation=parcellation,
        scale=scale,
        spatnull=spatnull,
        alpha=alpha,
        prob=prob
    )


def main():
    pvals, prob = [], []
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)
    for spatnull in simnulls.SPATNULLS:
        for alpha in simnulls.ALPHAS:
            if spatnull in simnulls.VERTEXWISE:
                pvals.append(
                    combine_nulls('vertex', 'fsaverage5', spatnull, alpha)
                )
                prob.append(
                    combine_shuffle('vertex', 'fsaverage5', spatnull, alpha)
                )
            for parcellation, annotations in parcellations.items():
                for scale in annotations:
                    pvals.append(
                        combine_nulls(parcellation, scale, spatnull, alpha)
                    )
                    prob.append(
                        combine_shuffle(parcellation, scale, spatnull, alpha)
                    )

    col = ['parcellation', 'scale', 'spatnull', 'alpha', 'sim', 'corr', 'pval']
    pvals = pd.concat(pvals, ignore_index=True)[col]
    pvals.to_csv(SIMDIR / 'pval_summary.csv', index=False)

    prob = pd.DataFrame(prob)[col[:4] + ['prob']]
    prob.to_csv(SIMDIR / 'prob_summary.csv', index=False)


if __name__ == "__main__":
    main()
