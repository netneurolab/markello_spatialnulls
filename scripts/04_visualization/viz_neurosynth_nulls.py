# -*- coding: utf-8 -*-
"""
Generates primary figures for NeuroSynth results
"""

from pathlib import Path
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 28.0

NSDIR = Path('./data/derivatives/neurosynth').resolve()
FIGDIR = Path('./figures/neurosynth').resolve()
METHODS = [
    'vazquez-rodriguez',
    'vasa',
    'hungarian',
    'baum',
    'cornblath',
    'burt2018',
    'burt2020',
    'moran',
]
NAIVE = [
    'naive-para',
    'naive-nonpara'
]
REP = {
    'atl-cammoun2012': 'scale500',
    'atl-schaefer2018': '1000Parcels7Networks'
}


def get_rect(bbox):
    """ Returns bounds of `bbox`
    """
    l, b = bbox[0]
    w, h = np.diff(bbox, axis=0)[0]

    return (l, b, w, h)


def make_lineplot(data, fname=None, **kwargs):
    """
    Generates lineplot of number of significant correlations for null methods

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with at least columns ['scale', 'n-sig', 'spintype'] for
        plotting
    fname : str or os.PathLike, optional
        Path to where generated figure should be saved
    kwargs : key-value pairs
        Passed to `ax.set()`

    Returns
    -------
    fig : matplotlib.figure.Figure
        Plotted figure
    """

    defaults = dict(xticklabels=[], xlabel='atlas resolution',
                    ylabel='# significant correlations', ylim=[0, 300])
    defaults.update(kwargs)

    # make plot of number of significant correlations remaining after
    # corection across resolutions of the given
    fig, ax1 = plt.subplots(1, 1)
    ax1 = sns.lineplot(x='scale', y='n_sig', hue='spintype', ax=ax1,
                       data=data, ci=None, estimator=None, linewidth=3,
                       legend=False)
    ax1.set(xticks=np.asarray(ax1.get_xticks()), **defaults)
    sns.despine(ax=ax1, offset=10, trim=True)

    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', transparent=True)
        plt.close(fig=fig)

    return fig


def make_distplot(methods, parcellation, fname=None):
    """
    Generates boxplot of null correlations for different null `methods`

    Parameters
    ----------
    methods : list of str
        List of methods for which to plot null distributions; each method will
        get its own box in the plot
    parcellation : {'atl-cammoun2012', 'atl-schaefer2018'}
        Parcellation for which to plot null distributions
    fname : str or os.PathLike, optional
        Path to where generated figure should be saved

    Returns
    -------
    fig : matplotlib.figure.Figure
        Plotted figure
    """

    # load the null correlations for the "representative" scale of the
    # specified parcellation
    df = pd.DataFrame(columns=['corr', 'method'])
    for sptype in methods:
        nulls = np.loadtxt(NSDIR / parcellation / 'nulls' / sptype
                           / f'{REP.get(parcellation)}_nulls.csv')
        df = df.append(pd.DataFrame({'corr': nulls}).assign(method=sptype))

    # we can have little a boxplot, as a treat
    fig, ax2 = plt.subplots(1, 1)
    ax2 = sns.boxplot(x='method', y='corr', data=df, ax=ax2, fliersize=0)
    ax2.set(xticklabels=[], yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            ylim=(0.05, 0.75),
            facecolor='none', ylabel='null correlations',
            xlabel='null framework')
    sns.despine(ax=ax2, offset=5, trim=True)
    patches = [
        f for f in ax2.get_children() if isinstance(f, mpatches.PathPatch)
    ]

    # alright so this is dumb, but basically this makes it so the shading
    # within each box plot represents the distribution of the data within that
    # range. it doesn't really work that well or look very good but OH WELL the
    # code is here so the garbage will do
    for n, patch in enumerate(patches):
        bbox = fig.transFigure.inverted().transform(patch.get_extents())
        sax = fig.add_axes(get_rect(bbox), zorder=-100)
        for spine in sax.spines.values():
            spine.set(linewidth=1.5, color='#3d3d3d')
        patch.remove()

        corr = np.asarray(df.query(f'method == "{methods[n]}"')['corr'])
        lo, hi = np.percentile(corr, [25, 75])
        x, y = sns.distributions._statsmodels_univariate_kde(
            corr, 'gau', 'scott', 100, 3, (-np.inf, np.inf)
        )
        idx, = np.where(np.logical_and(x >= lo, x < hi))
        cm = sns.light_palette(patch.get_facecolor(), as_cmap=True)
        sax.pcolormesh(y[:, None], cmap=cm, vmin=0, vmax=10,
                       rasterized=True)
        sax.set(xticks=[], yticks=[], ylim=(idx[0], idx[-1]))

    # save our lil baby frankenstein creation
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', transparent=True)
        plt.close(fig=fig)
        return

    return fig


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(NSDIR / 'summary.csv.gz')

    for parcellation in ['atl-cammoun2012', 'atl-schaefer2018']:
        dparc = data.query(f'parcellation == "{parcellation}"')
        order = sorted(dparc['scale'].unique(),
                       key=lambda x: int(re.search('(\d+)', x).group(1)))
        dparc = dparc.assign(
            scale=pd.Categorical(dparc['scale'], order, ordered=True)
        ).sort_values('scale')
        figdir = FIGDIR / parcellation
        figdir.mkdir(exist_ok=True, parents=True)

        fname = figdir / 'spatial_nulls_nsig.svg'
        make_lineplot(dparc.query(f'spintype in {METHODS}'), fname=fname)

        fname = figdir / 'naive_nulls_nsig.svg'
        make_lineplot(dparc.query(f'spintype in {NAIVE}'), fname=fname,
                      ylim=(450, 3550), yticks=range(500, 4000, 1000))

        # now make a plot of the distributions of null correlations from each
        # method for the highest resolution (1000 parcels, only comparable
        # scale across atlases)
        fname = figdir / 'spatial_nulls_dist.svg'
        make_distplot(METHODS, parcellation, fname)
