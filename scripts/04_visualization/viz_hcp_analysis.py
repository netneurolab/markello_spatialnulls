# -*- coding: utf-8 -*-
"""
Generates figures showing the analysis pipeline for the HCP / partition
specificity analyses in `03_results/run_hcp_nulls.py`
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from netneurotools import datasets as nndata
from parspin.partitions import YEO_CODES, RSN_AFILLIATION
from parspin.utils import save_brainmap

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Verdana']
plt.rcParams['font.size'] = 28.0

HCPDIR = Path('./data/derivatives/hcp').resolve()
ROIDIR = Path('./data/raw/rois').resolve()
SPINDIR = Path('./data/derivatives/spins').resolve()
FIGDIR = Path('./figures/hcp/analysis').resolve()
SPINTYPE = 'vazquez-rodriguez'
PARC, SCALE = 'atl-cammoun2012', 'scale125'
COLORS = np.array([
    np.array([0.7254902, 0.72941176, 0.73333333]),
    np.array([0.94117647, 0.40784314, 0.41176471])
])
YEOORDER = [
    'somatomotor',
    'visual',
    'dorsal attention',
    'frontoparietal',
    'limbic',
    'default mode',
    'ventral attention',
]


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # get T1/T2 for plotting
    path = HCPDIR / PARC / f'{SCALE}.csv'
    drop = [
        'lh_unknown', 'lh_corpuscallosum', 'rh_unknown', 'rh_corpuscallosum'
    ]
    t1t2 = pd.read_csv(path, index_col=0) \
             .assign(networks=RSN_AFILLIATION[PARC](SCALE).parcels) \
             .drop(drop)
    netmeans = t1t2.groupby('networks') \
                   .mean() \
                   .assign(network=YEO_CODES.keys())
    spins = np.loadtxt(SPINDIR / PARC / SPINTYPE / f'{SCALE}_spins.csv',
                       delimiter=',', dtype='int32', usecols=range(3))

    # get annotations
    fetcher = getattr(nndata, f'fetch_{PARC.split("-")[1]}')
    lh, rh = fetcher('fsaverage', data_dir=ROIDIR)[SCALE]

    # plot original
    data = np.asarray(t1t2['myelin'])
    save_brainmap(data, lh, rh, 't1wt2w_0.png',
                  colormap='coolwarm', vmin=1.0, vmax=1.6)
    for n, spin in enumerate(spins.T, 1):
        save_brainmap(data[spin], lh, rh, f't1wt2w_{n}.png',
                      colormap='coolwarm', vmin=1.0, vmax=1.6)

    null_dist = np.loadtxt(
        HCPDIR / PARC / 'nulls' / 'yeo' / SPINTYPE / f'{SCALE}_nulls.csv',
        delimiter=','
    )
    nulls = pd.DataFrame(dict(
        network=np.repeat(list(YEO_CODES.keys()), len(null_dist)),
        nulls=null_dist.flatten(order='F')
    ))

    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    ax = sns.boxplot(x='nulls', y='network', data=nulls, color='white',
                     fliersize=0, linewidth=1, order=YEOORDER, ax=ax)
    sns.despine(ax=ax, left=True)
    for i, art in enumerate(ax.artists):
        art.set_edgecolor('black')
        art.set_facecolor('none')
        for line in ax.lines[slice(i * 6, i * 6 + 6)]:
            line.set_color('black')
    order = [YEO_CODES[n] for n in YEOORDER]
    ax = sns.scatterplot(x='myelin', y='network', data=netmeans.loc[order],
                         ax=ax, color=np.array([112, 146, 255]) / 255,
                         zorder=10, s=75,)
    ax.set(xlabel='T1w/T2w', xticks=[1.0, 1.6], xlim=[1.0, 1.6])
    ax.figure.savefig(FIGDIR / f'null_distribution.svg', bbox_inches='tight',
                      transparent=True)
