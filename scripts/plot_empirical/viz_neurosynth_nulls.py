# -*- coding: utf-8 -*-
"""
Generates primary figures for NeuroSynth results
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from parspin.utils import SPATHUES

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 28.0

NSDIR = Path('./data/derivatives/neurosynth').resolve()
FIGDIR = Path('./figures/neurosynth').resolve()
METHODS = [
    'naive-para',
    'naive-nonpara',
    'vazquez-rodriguez',
    'baum',
    'cornblath',
    'vasa',
    'hungarian',
    'burt2018',
    'burt2020',
    'moran',
]
SPATIAL = METHODS[2:]
GRAY = np.asarray([175, 175, 175]) / 255


def make_lineplot(data, fname=None, line_kwargs=None, **kwargs):
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

    lp_kwargs = dict(ci=None, estimator=None, linewidth=3, legend=False)
    if line_kwargs is not None:
        lp_kwargs.update(line_kwargs)
    ax_kwargs = dict(xticklabels=[], xlabel='resolution',
                     ylabel='significant correlations remaining')
    ax_kwargs.update(kwargs)

    # make plot of number of significant correlations remaining after
    # corection across resolutions of the given
    fig, ax1 = plt.subplots(1, 1)
    ax1 = sns.lineplot(x='scale', y='n_sig', hue='spintype', ax=ax1,
                       data=data, **lp_kwargs)
    ax1.set(xticks=np.asarray(ax1.get_xticks()), **ax_kwargs)
    sns.despine(ax=ax1)

    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', transparent=True)
        plt.close(fig=fig)

    return fig


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(NSDIR / 'ns_summary.csv.gz')

    for parcellation in ['atl-cammoun2012', 'atl-schaefer2018']:
        dparc = data.query(f'parcellation == "{parcellation}"')
        order = sorted(dparc['scale'].unique(),
                       key=lambda x: int(re.search('(\d+)', x).group(1)))
        dparc = dparc.assign(
            scale=pd.Categorical(dparc['scale'], order, ordered=True)
        ).sort_values('scale')
        figdir = FIGDIR / parcellation
        figdir.mkdir(exist_ok=True, parents=True)

        fname = figdir / 'all_methods_nsig.svg'
        make_lineplot(dparc.query(f'spintype in {METHODS}'), fname=fname,
                      line_kwargs=dict(hue_order=METHODS,
                                       palette=SPATHUES[:2] + [GRAY] * 8),
                      ylim=(-50, 3550), yticks=np.arange(0, 4200, 700))

        fname = figdir / 'spatial_methods_nsig.svg'
        make_lineplot(dparc.query(f'spintype in {SPATIAL}'), fname=fname,
                      line_kwargs=dict(hue_order=SPATIAL,
                                       palette=SPATHUES[2:]),
                      ylim=(-5, 305), yticks=(0, 100, 200, 300), ylabel='')
