# -*- coding: utf-8 -*-
"""
Generates figures showing results from the simulated data
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # plot P(p < 0.05)
    figdir = FIGDIR / 'prob'
    figdir.mkdir(exist_ok=True)
    data = pd.read_csv(SIMDIR / 'prob_summary.csv')
    for parc, scale in PLOTS:
        plotdata = data.query(f'parcellation == "{parc}" & scale == "{scale}"')
        fg = sns.relplot(x='spatnull', y='prob', hue='spatnull', s=100,
                         col='alpha', data=plotdata, height=6, aspect=.75)
        fg.set_titles('{col_name}')
        fg.set(xlabel='', ylabel='P(p < 0.05)', xticklabels=[])
        xl = fg.axes[0, 0].get_xlim()
        for ax in fg.axes.flat:
            ax.hlines(0.05, *xl, linestyle='dashed', color='black')
        fg.savefig(figdir / f'{parc}.svg', bbox_inches='tight',
                   transparent=True)
        plt.close(fig=fg.fig)

    # plot n_nulls x p-value
    figdir = FIGDIR / 'pvals'
    figdir.mkdir(exist_ok=True)
    data = pd.read_csv(SIMDIR / 'pval_summary.csv')
    for parc, scale in PLOTS:
        plotdata = data.query(f'parcellation == "{parc}" & scale == "{scale}"')
        fg = sns.relplot(x='n_nulls', y='p_value', hue='spatnull', kind='line',
                         col='alpha', data=plotdata, height=6, aspect=0.75,
                         linewidth=2)
        fg.set_titles('{col_name}')
        fg.set(ylim=(fg.axes[0, 0].get_ylim()[0], 0.015), xticklabels=[],
               xlabel='', ylabel='Delta p (10k nulls)')
        fg.savefig(figdir / f'{parc}.svg', bbox_inches='tight',
                   transparent=True)
        plt.close(fig=fg.fig)

    # plot moran's I
    figdir = FIGDIR / 'moran'
    figdir.mkdir(exist_ok=True)
    data = pd.read_csv(SIMDIR / 'moran_summary.csv')
    _, scales = zip(*PLOTS)
    plotdata = data.query(f'scale in {scales}')
    for alpha in plotdata['alpha'].unique():
        adata = plotdata.query(f'alpha == "{alpha}"')
        ax = sns.boxplot(x='spatnull', y='moran', hue='parcellation',
                         data=adata, fliersize=0, linewidth=1.0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.legend_.set_visible(False)
        ax.set(ylabel="moran's i", xlabel='null framework')
        sns.despine(ax=ax)
        ax.figure.savefig(figdir / f'{alpha}.svg', bbox_inches='tight',
                          transparent=True)
        plt.close(fig=ax.figure)

    # plot correlations (?)
    figdir = FIGDIR / 'corrs'
    figdir.mkdir(exist_ok=True)
    data = pd.read_csv(SIMDIR / 'corr_summary.csv')
    _, scales = zip(*PLOTS)
    plotdata = data.query(f'scale in {scales}')
    for alpha in plotdata['alpha'].unique():
        adata = plotdata.query(f'alpha == "{alpha}"')
        ax = sns.kdeplot(x='corr', hue='parcellation', data=adata,
                         legend=False)
        ax.set(xlim=[-0.05, 0.35], xticks=[0, 0.15, 0.30])
        sns.despine(ax=ax)
        ax.figure.savefig(figdir / f'{alpha}.svg', bbox_inches='tight',
                          transparent=True)
        plt.close(fig=ax.figure)
