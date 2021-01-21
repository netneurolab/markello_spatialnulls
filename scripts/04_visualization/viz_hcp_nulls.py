# -*- coding: utf-8 -*-
"""
Generates primary figures for HCP results
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 28.0

HCPDIR = Path('./data/derivatives/hcp').resolve()
FIGDIR = Path('./figures/hcp').resolve()
COLORS = np.array([
    np.array([0.7254902, 0.72941176, 0.73333333]),
    np.array([0.94117647, 0.40784314, 0.41176471])
])
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
YEOORDER = [
    'somatomotor',
    'visual',
    'dorsal attention',
    'frontoparietal',
    'limbic',
    'default mode',
    'ventral attention',
]
VEKORDER = [
    'primary sensory cortex',
    'primary motor cortex',
    'primary/secondary sensory',
    'association cortex 2',
    'insular cortex',
    'association cortex 1',
    'limbic regions'
]


def make_barplot(data, netorder, methods=None, fname=None, **kwargs):
    """
    Makes barplot of network z-scores as a function of null model

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with as least columns ['zscore', 'network', 'parcellation',
        'scale', 'sig']
    netorder : list
        Order in which networks should be plotted within each barplot
    methods : list, optional
        Null methods that should be plotted (and the order in which they should
        be plotted)
    fname : str or os.PathLike, optional
        Path to where generated figure should be saved
    kwargs : key-value pairs
        Passed to `ax.set()` on the generated boxplot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Plotted figure
    """

    defaults = dict(ylabel='', xticklabels=[], xticks=[], ylim=(-3.5, 3.5))
    defaults.update(kwargs)

    if methods is None:
        methods = data['spintype'].unique()
    fig, axes = plt.subplots(1, len(methods), sharey=True,
                             figsize=(3.125 * len(methods), 4))

    # edge case for if we're only plotting one barplot
    if not isinstance(axes, (np.ndarray, list)):
        axes = [axes]

    for n, ax in enumerate(axes):
        # get the data for the relevant null
        d = data.query(f'spintype == "{methods[n]}"')
        palette = COLORS[np.asarray(d['sig'], dtype='int')]
        # plot!
        ax = sns.barplot('network', 'zscore', data=d,
                         order=netorder, palette=palette, ax=ax)
        lab = '-\n'.join(methods[n].split('-'))
        ax.set(xlabel=lab, **defaults)
        sns.despine(ax=ax, bottom=True)
        ax.hlines(0, -0.5, len(netorder) - 0.5, linewidth=0.5)
    axes[0].set_ylabel('T1w/T2w (z)', labelpad=10)

    if fname is not None:
        fname.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(fname, bbox_inches='tight', transparent=True)
        plt.close(fig=fig)

    return fig


def make_heatmap(data, order, fname=None, **kwargs):
    """
    Makes heatmap plotting network z-scores as a function of atlas/resolution

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with as least columns ['zscore', 'network', 'parcellation',
        'scale', 'sig']
    order : list
        Order in which networks should be plotted
    fname : str or os.PathLike, optional
        Path to where generated figure should be saved
    kwargs : key-value pairs
        Passed to `sns.heatmap`

    Returns
    -------
    fig : matplotlib.figure.Figure
        Plotted figure
    """
    defaults = dict(
        cmap='coolwarm', vmin=-2.5, center=0, vmax=2.5, xticklabels=[],
        yticklabels=[]
    )
    defaults.update(kwargs)
    cbar_kws = dict(
        ticks=[defaults['vmin'], defaults['center'], defaults['vmax']]
    )

    zscores = pd.pivot_table(data, values='zscore',
                             columns='network',
                             index=['parcellation', 'scale'])[order]
    sig = pd.pivot_table(data, values='sig',
                         columns='network',
                         index=['parcellation', 'scale'])[order]

    fig, ax = plt.subplots(1, 1)
    ax = sns.heatmap(zscores, mask=np.logical_not(sig), alpha=1.0,
                     cbar_kws=cbar_kws, **defaults)
    ax = sns.heatmap(zscores, mask=sig, alpha=0.3, cbar=False, **defaults)
    ax.set(ylim=[15, 0], yticks=[], xticks=[], ylabel='',
           xlabel='')
    ax.hlines(5, *ax.get_xlim())

    if fname is not None:
        fname.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(fname, bbox_inches='tight', transparent=True)
        plt.close(fig=fig)
        return

    return fig


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(HCPDIR / 'summary.csv')
    data = data.assign(sig=data['pval'] < 0.05,
                       scale=pd.Categorical(data['scale'],
                                            data['scale'].unique(),
                                            ordered=True))

    # plot parcellation results
    for netclass, netorder in zip(['yeo', 'vek'], [YEOORDER, VEKORDER]):
        for parc in ['atl-cammoun2012', 'atl-schaefer2018']:
            scales = data.query(f'parcellation == "{parc}"')['scale'].unique()
            for scale in sorted(scales):
                fname = FIGDIR / parc / netclass / f'{scale}.svg'

                # extract relevant portion of dataframe we want to plot
                dparc = data.query(f'parcellation == "{parc}" '
                                   f'& netclass == "{netclass}" '
                                   f'& scale == "{scale}" '
                                   f'& spintype in {METHODS}')
                dparc = dparc.assign(
                    network=pd.Categorical(dparc['network'], netorder,
                                           ordered=True)
                ).sort_values('network')
                fig = make_barplot(dparc, netorder, METHODS, fname=fname,
                                   yticks=[-2, 0, 2])

                # okay, now do the same for the naive models (separately)
                for sptype in NAIVE:
                    fname = (FIGDIR / parc / netclass / 'naive'
                             / f'{scale}_{sptype}.svg')
                    dparc = data.query(f'parcellation == "{parc}" '
                                       f'& netclass == "{netclass}" '
                                       f'& scale == "{scale}" '
                                       f'& spintype == "{sptype}"')
                    dparc = dparc.assign(
                        network=pd.Categorical(dparc['network'], netorder,
                                               ordered=True)
                    ).sort_values('network')
                    fig = make_barplot(dparc, netorder, ylim=None)
                    ax = fig.axes[0]
                    if sptype == 'naive-para':
                        ax.set_ylabel('T1w/T2w', labelpad=10)
                        ax.set(ylim=(-2, 2), yticks=[-2, 0, 2])
                    else:
                        ylim = max([abs(f) for f in ax.get_ylim()])
                        ylim += 5 - (ylim % 5)
                        ax.set(ylim=(-ylim, ylim), yticks=[-ylim, 0, ylim])
                    fname.parent.mkdir(exist_ok=True, parents=True)
                    fig.savefig(fname, bbox_inches='tight', transparent=True)
                    plt.close(fig=fig)

        for method in METHODS + NAIVE:
            dparc = data.query('parcellation != "vertex" '
                               f'& netclass == "{netclass}" '
                               f'& spintype == "{method}"')
            kwargs = {}
            if method == 'naive-para':
                kwargs = {'vmin': -1, 'vmax': 1}
            elif method == 'naive-nonpara':
                kwargs = {'vmin': -7.5, 'vmax': 7.5}
            fname = FIGDIR / netclass / f'{method}.svg'
            make_heatmap(dparc, netorder, fname=fname, **kwargs)
