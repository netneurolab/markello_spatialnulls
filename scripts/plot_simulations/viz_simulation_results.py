# -*- coding: utf-8 -*-
"""
Generates figures showing results from the simulated data
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from netneurotools import stats as nnstats
from parspin.plotting import savefig
from parspin.simnulls import load_vertex_data, SPATNULLS
from parspin.utils import PARCHUES, SPATHUES

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 28.0

SIMDIR = Path('./data/derivatives/simulated').resolve()
FIGDIR = Path('./figures/simulated').resolve()
PLOTS = (
    ('vertex', 'fsaverage5'),
    ('atl-cammoun2012', 'scale500'),
    ('atl-schaefer2018', '1000Parcels7Networks')
)
PARCS, SCALES = zip(*PLOTS)
REPLACE = {
    'atl-cammoun2012': 'cammoun',
    'atl-schaefer2018': 'schaefer'
}


def remove_caps(ax):
    # remove caps on boxplot whiskers
    caps = (list(range(2, len(ax.lines), 6))
            + list(range(3, len(ax.lines), 6)))
    for line in caps:
        ax.lines[line].set_visible(False)


if __name__ == "__main__":
    data = pd.read_csv(SIMDIR / 'pval_summary.csv.gz')
    data = data.query(f'scale in {SCALES}')
    with np.errstate(divide='ignore'):
        logpval = np.asarray(-np.log10(data['pval']))
    logpval[np.isinf(logpval)] = logpval[~np.isinf(logpval)].max()
    data.loc[data.index, '-log10(p)'] = logpval

    # plot -log10(p) vs alpha
    fg = sns.relplot(x='alpha', y='-log10(p)', hue='spatnull',
                     hue_order=SPATNULLS, palette=SPATHUES, linewidth=2.5,
                     col='parcellation', kind='line', data=data)
    fg.set_titles('{col_name}')
    fg.set(xticklabels=[0.0, '', '', 1.5, '', '', 3.0],
           xlabel='spatial autocorrelation',
           ylim=(0, 3.5))
    xl = fg.axes[0, 0].get_xlim()
    for ax in fg.axes.flat:
        ax.hlines(-np.log10(0.05), *xl, linestyle='dashed', color='black',
                  linewidth=1.0)
    savefig(fg.fig, FIGDIR / 'pvals' / 'pvals_all.svg')

    # plot -log10(p) vs alpha for simulations with r = 0.15 +/- 0.025
    fg = sns.relplot(x='alpha', y='-log10(p)', hue='spatnull',
                     hue_order=SPATNULLS, palette=SPATHUES, linewidth=2.5,
                     col='parcellation', kind='line',
                     data=data.query('corr >= 0.125 & corr <= 0.175'))
    fg.set_titles('{col_name}')
    fg.set(xticklabels=[0.0, '', '', 1.5, '', '', 3.0],
           xlabel='spatial autocorrelation',
           ylim=(0, 3.5))
    xl = fg.axes[0, 0].get_xlim()
    for ax in fg.axes.flat:
        ax.hlines(-np.log10(0.05), *xl, linestyle='dashed', color='black',
                  linewidth=1.0)
    savefig(fg.fig, FIGDIR / 'pvals' / 'pvals_thresh.svg')

    # plot correlations for dense v parcellated
    ax = sns.boxplot(x='alpha', y='corr', hue='parcellation',
                     data=data.query('spatnull == "naive-nonpara"'),
                     palette=PARCHUES, fliersize=0, saturation=1,
                     linewidth=1.0)
    ax.legend_.set_visible(False)
    ax.set(ylim=(0.025, 0.275), yticks=[0.05, 0.15, 0.25],
           xticklabels=[0.0, '', '', 1.5, '', '', 3.0],
           xlabel='spatial autocorrelation',
           ylabel='correlation (r)')
    remove_caps(ax)
    sns.despine(ax=ax)
    savefig(ax.figure, FIGDIR / 'corrs' / 'all_corrs.svg')

    # plot moran's I for diff spatnulls
    data = pd.read_csv(SIMDIR / 'moran_summary.csv.gz')
    data = data.query(f'scale in {SCALES}')
    for ialpha, alpha in enumerate(data['alpha'].unique()):
        # get relevant data and calculate delta Moran's I (null - empirical)
        adata = data.query(f'alpha == "{alpha}"')
        plotdata = adata.query('spatnull != "empirical"')
        empirical = adata.query('spatnull == "empirical" & sim == 9999') \
                         .set_index('parcellation') \
                         .loc[np.asarray(plotdata['parcellation']), 'moran']
        plotdata = plotdata.assign(**{
            'd(moran)': np.asarray(plotdata['moran']) - np.asarray(empirical)
        })
        # make boxplot
        ax = sns.boxplot(x='spatnull', y='d(moran)', hue='parcellation',
                         data=plotdata, fliersize=0, linewidth=1.0,
                         order=SPATNULLS[1:], palette=PARCHUES,
                         saturation=1)
        ax.legend_.set_visible(False)
        ax.hlines(0.0, *ax.get_xlim(), linestyle='dashed', color='black',
                  linewidth=1.0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set(xlabel='null framework')
        sns.despine(ax=ax)
        # remove caps on boxplot whiskers
        remove_caps(ax)
        # add empirical Moran's I as text to plot
        text = pd.DataFrame(
            np.unique(empirical.reset_index()
                               .replace(REPLACE)
                               .to_records(index=False))
        ).set_index('parcellation')
        ystart = 0.25 if ialpha >= 3 else 0.9
        for n, parc in enumerate(['vertex', 'cammoun', 'schaefer']):
            ax.text(0.25, ystart - (0.1 * n),
                    f'{parc}: {text.loc[parc, "moran"]:.2f}',
                    color=PARCHUES[n],
                    transform=ax.transAxes,
                    fontdict={'fontsize': 24})
        savefig(ax.figure, FIGDIR / 'moran' / f'null_{alpha}.svg')

        # now, boxplot of empirical Moran's I across ALL simulations
        ax = sns.boxplot(x='spatnull', y='moran', hue='parcellation',
                         data=adata, fliersize=0, linewidth=1.0,
                         order=['empirical'] + SPATNULLS[1:], palette=PARCHUES,
                         saturation=1)
        ax.legend_.set_visible(False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set(xlabel='null framework')
        sns.despine(ax=ax)
        # remove caps on boxplot whiskers
        remove_caps(ax)
        savefig(ax.figure, FIGDIR / 'moran' / f'empirical_{alpha}.svg')

    # plot P(p < 0.05)
    data = pd.read_csv(SIMDIR / 'prob_summary.csv.gz')
    data = data.query(f'scale in {SCALES}')
    fg = sns.relplot(x='alpha', y='prob', hue='spatnull', col='parcellation',
                     data=data, hue_order=SPATNULLS, palette=SPATHUES,
                     kind='line', linewidth=2.5)
    fg.set_titles('{col_name}')
    fg.legend.set_visible(False)
    fg.set(xticklabels=[0.0, '', '', 1.5, '', '', 3.0],
           xlabel='spatial autocorrelation',
           ylabel='Prob(p < 0.05)',
           ylim=(0, 1.0), yticks=[0, 0.25, 0.5, 0.75, 1.0])
    xl = fg.axes[0, 0].get_xlim()
    for ax in fg.axes.flat:
        ax.hlines(0.05, *xl, linestyle='dashed', color='black',
                  linewidth=1.0)
    savefig(fg.fig, FIGDIR / 'prob' / 'all_probp05.svg')

    # now plot P(p < 0.05) as a function of parcellation resolution
    data = pd.read_csv(SIMDIR / 'prob_summary.csv.gz')
    for parc in ('atl-cammoun2012', 'atl-schaefer2018'):
        plotdata = data.query(f'parcellation == "{parc}"')
        plotdata = plotdata.assign(
            scale=plotdata['scale'].apply(
                lambda x: re.search('(\d+)', x).group(1)
            )
        )
        scales = np.asarray(plotdata['scale'].unique())
        fg = sns.relplot(x='scale', y='prob', hue='spatnull',
                         col='alpha', data=plotdata,
                         hue_order=SPATNULLS, palette=SPATHUES,
                         kind='line', linewidth=2.5)
        fg.legend.set_visible(False)
        fg.set_titles('{col_name}')
        fg.set(xticklabels=[], xticks=scales,
               xlabel='resolution', ylabel='Prob(p < 0.05)',
               ylim=(0, 1.0), yticks=[0, 0.25, 0.5, 0.75, 1.0])
        xl = fg.axes[0, 0].get_xlim()
        for ax in fg.axes.flat:
            ax.hlines(0.05, *xl, linestyle='dashed', color='black',
                      linewidth=1.0)
        savefig(fg.fig, FIGDIR / 'prob' / f'{parc}_probp05.svg')

    # plot shuffled correlation distributions as func of SA (vertex only)
    data = pd.DataFrame(columns=['alpha', 'corrs'])
    alphas = ['alpha-0.0', 'alpha-1.0', 'alpha-2.0', 'alpha-3.0']
    for alpha in alphas:
        x, y = load_vertex_data(SIMDIR / alpha, n_sim=1000)
        y = y[:, np.random.default_rng(1).permutation(1000)]
        corrs = nnstats.efficient_pearsonr(x, y, nan_policy='omit')[0]
        data = data.append(pd.DataFrame({'alpha': alpha, 'corrs': corrs}),
                           ignore_index=True)
    colors = np.asarray(sns.color_palette('Greens', 10, desat=0.5))
    ax = sns.kdeplot(x='corrs', hue='alpha', data=data, legend=False,
                     palette=list(colors[[2, 4, 6, 8]]), hue_order=alphas,
                     clip=[-1, 1], linewidth=3.0)
    ax.set(xlim=(-1.05, 1.05), xticks=np.arange(-1, 1.5, 0.5),
           xticklabels=[-1, '', 0, '', 1],
           ylim=(-.05, 14), yticks=[0, 7, 14],
           xlabel='correlation (r) between\nrandom simulated maps',
           ylabel='density')
    sns.despine(ax=ax)
    savefig(ax.figure, FIGDIR / 'prob' / 'vertex_corrs.svg')
