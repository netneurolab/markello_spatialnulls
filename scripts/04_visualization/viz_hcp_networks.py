# -*- coding: utf-8 -*-
"""
Generates primary figures for HCP results
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
import seaborn as sns

from parspin.partitions import NET_OPTIONS, YEO_CODES
from parspin.plotting import savefig
from parspin import utils as putils

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 20.0
plt.rcParams['axes.titlesize'] = 'medium'

HCPDIR = Path('./data/derivatives/hcp').resolve()
FIGDIR = Path('./figures/hcp').resolve()
THRESH = 'thresh100'
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
SPATHUES = putils.SPATHUES[2:]
NETWORKS = ['visual', 'ventral attention']
PARC, SCALE = 'atl-cammoun2012', 'scale500'


def load_data(netclass, parc, scale):
    """
    Parameters
    ----------
    netclass : {'yeo', 'vek'}
        Network classes to use
    parc : {'atl-cammoun2012', 'atl-schaefer2018', 'vertex'}
        Name of parcellation to use
    scale : str
        Scale of parcellation to use. Must be valid scale for specified `parc`

    Returns
    -------
    data : pandas.DataFrame
        Loaded dataframe with columns 'myelin' and 'networks'
    """

    # load data for provided `parcellation` and `scale`
    data = pd.read_csv(HCPDIR / parc / f'{scale}.csv', index_col=0)

    # get the RSN affiliations for the provided parcellation + scale
    networks = NET_OPTIONS[netclass][parc](scale)
    if parc == 'vertex':
        # we want the vertex-level affiliations if we have vertex data
        data = data.assign(networks=getattr(networks, 'vertices'))
        # when working with vertex-level data, our spins were generated with
        # the medial wall / corpuscallosum included, but we need to set these
        # to NaN so they're ignored in the final sums
        data.loc[data['networks'] == 0, 'myelin'] = np.nan
    else:
        # get the parcel-level affiliations if we have parcellated data
        data = data.assign(networks=getattr(networks, 'parcels'))
        # when working with parcellated data, our spins were NOT generated with
        # the medial wall / corpuscallosum included, so we should drop these
        # parcels (which should [ideally] have values of ~=0)
        todrop = np.array(putils.DROP)[np.isin(putils.DROP, data.index)]
        if len(todrop) > 0:
            data = data.drop(todrop, axis=0)

    return data


if __name__ == "__main__":
    data = load_data('yeo', PARC, SCALE)
    netmeans = ndimage.mean(data['myelin'], data['networks'],
                            np.unique(data['networks']))

    fig, axes = plt.subplots(len(NETWORKS), len(METHODS),
                             sharex=True, sharey=True, figsize=(25, 10))
    for row, network in enumerate(NETWORKS):
        for col, spatnull in enumerate(METHODS):
            idx = YEO_CODES[network] - 1
            ax = axes[row, col]
            fn = (HCPDIR / PARC / 'nulls' / 'yeo' / spatnull / THRESH
                  / f'{SCALE}_nulls.csv')
            null = np.loadtxt(fn, delimiter=',')[:, idx]
            sns.kdeplot(null, fill=True, alpha=0.4, color=SPATHUES[col],
                        ax=ax)
            sns.despine(ax=ax)
            if row == 0:
                ax.set_title(spatnull.replace('-', '-\n'), y=1.02)
            if col == 0:
                ax.set_ylabel(f'density\n({network})')
            ax.vlines(netmeans[idx], 0, 12.5, linestyle='dashed',
                      linewidth=1.0)

    ax.set(xlim=(1.0, 1.6), ylim=(0, 15),
           xticks=np.arange(1, 1.75, 0.15), yticks=np.arange(0, 20, 5),
           xticklabels=(1.0, None, 1.3, None, 1.6),
           yticklabels=(0, None, None, 15))

    fn = FIGDIR / THRESH / 'supplementary' / 'null_examples.svg'
    savefig(fig, fn)
