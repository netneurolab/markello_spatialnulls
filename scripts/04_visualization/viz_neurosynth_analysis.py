# -*- coding: utf-8 -*-
"""
Generates figures showing the analysis pipeline for the NeuroSynth analyses in
`03_results/run_neurosynth_nulls.py`
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import seaborn as sns

from netneurotools import datasets as nndata
from parspin.plotting import save_brainmap
from parspin.utils import PARULA

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 28.0

FIGSIZE = 500
NSDIR = Path('./data/derivatives/neurosynth').resolve()
ROIDIR = Path('./data/raw/rois').resolve()
SPINDIR = Path('./data/derivatives/spins').resolve()
FIGDIR = Path('./figures/neurosynth/analysis').resolve()
SPINTYPE = 'vazquez-rodriguez'
PARC, SCALE = 'atl-cammoun2012', 'scale125'


def plot_save_heatmap(data, fname=None, diagonal=False, cbar=False,
                      cbar_label=None, imshow_kwargs=None, cbar_kwargs=None,
                      aspect=None):
    """
    Plots provided `data` as heatmap and optionally saves to `fname`

    Parameters
    ----------
    data : array_like
        Data to be plotted as a heatmap
    fname : str or os.PathLike, optional
        Path to where generated figure should be saved. If not specified,
        figure will not be saved. Default: None
    diagonal : bool, optional
        Whether to plot the diagonal. Default: False
    cbar : bool, optional
        Whether to plot the colorbar. Default: False
    cbar_label : str, optional
        Label for the colorbar, if `cbar` is True. Default: Nne
    imshow_kwargs : dict, optional
        Key-value pairs passed to `matplotlib.pyplot.imshow()`
    cbar_kwargs : dict, optional
        Key-value pairs passed to `matplotlib.figure.Figure.colorbar()`
    aspect : str or float, optional
        Aspect parameter for plotted heatmap. Default: None

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Plotted figure
    """

    # mask diagonal
    plot = data.copy()
    if not diagonal:
        np.fill_diagonal(plot, np.nan)

    fig, ax = plt.subplots(1, 1)

    imshow_opts = dict(cmap=PARULA, rasterized=True)
    if imshow_kwargs is not None:
        imshow_opts.update(imshow_kwargs)
    coll = ax.imshow(plot, **imshow_opts)
    ax.set(xticks=[], yticks=[])

    if aspect is not None:
        ax.set_aspect(aspect)

    # remove outlines around heatmap
    for sp in ax.spines.values():
        sp.set_visible(False)

    # add colorbar
    if cbar:
        cbar_opts = dict(drawedges=False, ticks=[], shrink=1.0)
        if cbar_kwargs is not None:
            cbar_opts.update(cbar_kwargs)
        cbar = fig.colorbar(coll, **cbar_opts)
        cbar.outline.set_visible(False)
        if cbar_label is not None:
            cbar.set_label(cbar_label, rotation=270, fontsize=18, labelpad=25)

    # save to file, if desired
    if fname is not None:
        out = fname.split('.')[0] + '.svg'
        fig.savefig(FIGDIR / out, bbox_inches='tight', transparent=True)

    return fig


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)

    fetcher = getattr(nndata, f'fetch_{PARC.split("-")[1]}')
    lh, rh = fetcher('fsaverage', data_dir=ROIDIR)[SCALE]
    data = pd.read_csv(NSDIR / PARC / f'{SCALE}.csv', index_col=0).drop(
        ['lh_unknown', 'lh_corpuscallosum', 'rh_unknown', 'rh_corpuscallosum']
    )
    spins = np.loadtxt(SPINDIR / PARC / SPINTYPE / f'{SCALE}_spins.csv',
                       delimiter=',', dtype='int32', usecols=range(5))

    cmap = sns.blend_palette((PARULA(20), [1, 1, 1], PARULA(215)),
                             as_cmap=True)

    corr_orig = np.corrcoef(data.T)
    linkage = hierarchy.linkage(corr_orig, optimal_ordering=True)
    inds = hierarchy.leaves_list(linkage)
    all_data = [np.asarray(data)] + [
        np.asarray(data)[spin] for spin in spins.T
    ]
    terms = ['salience', 'fear']

    for n, cdata in enumerate(all_data):
        # save region-by-term data matrix
        ratio = cdata.shape[-1] / cdata.shape[0]
        plot_save_heatmap(cdata[:, inds].T, fname=f'region_by_term_{n}',
                          diagonal=True, cbar=True, cbar_label='zscore',
                          imshow_kwargs=dict(vmin=-2.5, vmax=4, cmap=cmap),
                          cbar_kwargs=dict(shrink=0.6), aspect=1 / ratio ** 2)

        # save term-by-term correlation matrix
        corr = np.corrcoef(data.T, cdata.T)[123:, :123]
        plot_save_heatmap(corr[np.ix_(inds, inds)], fname=f'corrmat_{n}',
                          diagonal=False, cbar=True,
                          cbar_label='correlation (r)',
                          imshow_kwargs=dict(vmin=-0.5, vmax=1, cmap=cmap))

    # save resampling arrays
    plot_save_heatmap(np.column_stack([range(len(data)), spins]),
                      fname='resamples', diagonal=True, cbar=False,
                      aspect='auto',
                      imshow_kwargs=dict(cmap=cmap))

    # save sample brain maps
    for term in terms:
        # symmetrical colorbar
        vmin, vmax = np.percentile(data[term], [2.5, 97.5])
        lim = max(abs(vmin), abs(vmax))
        save_brainmap(data[term], f'{term}_0.png', lh=lh, rh=rh,
                      colormap=cmap, vmin=-lim, vmax=lim)
        # get spins, in case we want those, too
        for n, spin in enumerate(spins.T, 1):
            plot = np.asarray(data[term])[spin]
            save_brainmap(plot, f'{term}_{n}.png', lh=lh, rh=rh,
                          colormap=cmap, vmin=-lim, vmax=lim)

    # save example null distribution
    null_dist = np.loadtxt(
        NSDIR / PARC / 'nulls' / SPINTYPE / f'{SCALE}_nulls.csv'
    )
    fig, ax = plt.subplots(1, 1)
    ax = sns.kdeplot(null_dist, color=PARULA(20), linewidth=3, shade=True)
    sns.despine(ax=ax, left=True)
    ax.set(yticks=[], xlim=(0.4, 1.1), xticks=(0.5, 0.75, 1.0))
    (x, y), = np.where(np.isin(data.columns, terms))
    ax.vlines(corr_orig[x, y], *ax.get_ylim(), linestyle='dashed')
    ax.figure.savefig(FIGDIR / 'nulldist.svg', bbox_inches='tight',
                      transparent=True)
