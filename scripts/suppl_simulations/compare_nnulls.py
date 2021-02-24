# -*- coding: utf-8 -*-
"""
Script for assessing how the number of nulls used to generate a p-value
influences the p-value
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from netneurotools import stats as nnstats
from parspin import simnulls, utils as putils
from parspin.plotting import savefig

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 20.0

ROIDIR = Path('./data/raw/rois').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()
OUTDIR = Path('./data/derivatives/supplementary/comp_nnulls')
FIGDIR = Path('./figures/supplementary/comp_nnulls')

SEED = 1234  # reproducibility
SIM = 9999  # which simulation was used to generate 10000 nulls
N_PVALS = 1000  # how many repeated draws should be done to calculate pvals
PLOTS = (
    ('vertex', 'fsaverage5'),
    ('atl-cammoun2012', 'scale500'),
    ('atl-schaefer2018', '1000Parcels7Networks')
)
PARCS, SCALES = zip(*PLOTS)


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

    df = pd.melt(pd.DataFrame(pvals), var_name='n_nulls', value_name='d(pval)')
    # add single p-value generated from 10000 nulls
    df = df.assign(
        parcellation=parcellation,
        scale=scale,
        spatnull=spatnull,
        alpha=alpha
    )

    return df[
        'parcellation', 'scale', 'spatnull', 'alpha', 'n_nulls', 'd(pval)'
    ]


def run_analysis():
    """ Runs p-value x n_nulls analysis

    Returns
    -------
    pvals : pd.DataFrame
        Data examining p-values based on number of nulls used
    """

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fn = OUTDIR / 'nnulls_summary.csv'
    if fn.exists():
        return pd.read_csv(fn)

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
    subsets.to_csv(OUTDIR / 'nnulls_summary.csv.gz', index=False)
    return subsets


if __name__ == "__main__":
    data = run_analysis()
    palette = dict(zip(simnulls.SPATNULLS, putils.SPATHUES))
    for parc, scale in PLOTS:
        plotdata = data.query(f'parcellation == "{parc}" & scale == "{scale}"')
        fg = sns.relplot(x='n_nulls', y='d(pval)', hue='spatnull', col='alpha',
                         hue_order=simnulls.SPATNULLS, data=plotdata,
                         palette=palette, kind='line', ci=95)
        fg.set_titles('{col_name}')
        fg.axes[0, 0].set_xscale('log')
        fg.set(xlim=(75, 6000))
        savefig(fg.fig, FIGDIR / f'{scale}.svg')
