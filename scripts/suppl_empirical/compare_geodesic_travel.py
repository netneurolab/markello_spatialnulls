# -*- coding: utf-8 -*-
"""
Compares T1w/T2w surrogates generated w/ and w/o medial wall travel
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from brainsmash.mapgen import Base
from brainspace.null_models import MoranRandomization
from netneurotools import stats as nnstats
from parspin import burt, plotting, utils as putils

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']
plt.rcParams['font.size'] = 28.0

SEED = 1234
N_SURROGATES = 1000
N_PROC = 12
ROIDIR = Path('./data/raw/rois').resolve()
HCPDIR = Path('./data/derivatives/hcp').resolve()
DISTDIR = Path('./data/derivatives/geodesic')
FIGDIR = Path('./figures/supplementary/comp_geodesic')
OUTDIR = Path('./data/derivatives/supplementary/comp_geodesic')
METHODS = [
    'burt2018',
    'burt2020',
    'moran'
]


def get_distcorr_stats(dist, seed=SEED):
    """
    Gets statistics from distribution of correlations `corr`

    Uses r-to-z transform (and inverse) to compute stats on z-transformed data

    Parameters
    ----------
    dist : array_like
        Distribution of correlations (r)
    seed : int, optional
        Seed for random generation of bootstraps

    Returns
    -------
    mean : float
        Average correlation of `dist`
    ci : tuple-of-float
        (2.5%, 97.5%) confidence intervals on `mean` correlation
    """

    dist_z = np.arctanh(dist)
    rs = np.random.default_rng(seed)
    dist_ci = [
        rs.choice(dist_z, size=len(dist_z)).mean() for _ in range(10000)
    ]
    ci = np.tanh(np.percentile(dist_ci, [2.5, 97.5]))
    mean = np.tanh(dist_z.mean())

    return (mean, tuple(ci))


if __name__ == "__main__":
    FIGDIR.mkdir(exist_ok=True, parents=True)
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    for name, annotations in parcellations.items():
        print(f'PARCELLATION: {name}')
        for scale, annot in annotations.items():
            print(f'Comparing surrogates for {scale}')

            # load T1w/T2w for given parcellation + resolution
            data = pd.read_csv(HCPDIR / name / f'{scale}.csv', index_col=0)
            data = data.drop([i for i in data.index if i in putils.DROP])
            data = data['myelin']

            # generate surrogates for each method using distance matrix w/ and
            # w/o medial wall travel (one surrogate per method per dist matrix)
            burt2018, burt2020, moran = [], [], []
            for med in [True, False]:
                for method, surrs in zip(METHODS, [burt2018, burt2020, moran]):
                    surrdata = []
                    for hd, dist, _ in putils.yield_data_dist(DISTDIR, name,
                                                              scale, data,
                                                              medial=med,
                                                              inverse=False):
                        if method == 'burt2018':
                            surr = burt.batch_surrogates(dist, hd, seed=SEED,
                                                         n_surr=N_SURROGATES,
                                                         n_jobs=N_PROC).T
                        elif method == 'burt2020':
                            base = Base(hd, dist, resample=True, seed=SEED,
                                        n_jobs=N_PROC)
                            surr = base(N_SURROGATES, 50)
                        elif method == 'moran':
                            np.fill_diagonal(dist, 1)
                            dist **= -1
                            mrs = MoranRandomization(n_rep=N_SURROGATES,
                                                     tol=1e-6,
                                                     random_state=SEED)
                            surr = mrs.fit(dist).randomize(hd)
                        else:
                            raise ValueError('Your method name has a typo.')
                        surrdata.append(np.atleast_2d(surr).T)
                    surrs.append(np.row_stack(surrdata))

            # now we need to save all the surrogate brain maps and make some
            # scatterplots comparing them to one another
            for method, surrs in zip(METHODS, [burt2018, burt2020, moran]):
                figdir = FIGDIR / name / method
                figdir.mkdir(exist_ok=True, parents=True)

                # brain maps of example brains
                vmin, vmax = np.percentile(surrs, [2.5, 97.5])
                for n, medial in enumerate(['medial', 'nomedial']):
                    fn = figdir / medial / f'{scale}.png'
                    if fn.exists():
                        continue
                    plotting.save_brainmap(surrs[n][:, 0], fn,
                                           annot.lh, annot.rh,
                                           subject_id='fsaverage5',
                                           colormap='coolwarm', colorbar=False,
                                           vmin=vmin, vmax=vmax,
                                           views=['lat', 'med'])

                # scatter plot of example brains
                fig, ax = plt.subplots(1, 1)
                ax.scatter(surrs[0][:, 0], surrs[1][:, 0], s=75,
                           edgecolor=np.array([60, 60, 60]) / 255,
                           facecolor=np.array([223, 121, 122]) / 255)
                for side in ['right', 'top']:
                    ax.spines[side].set_visible(False)
                ax.set(xlabel='with medial wall', ylabel='without medial wall')
                l, h = ax.get_xlim()
                ax.plot([l, h], [l, h], zorder=0)
                ax.figure.savefig(figdir / f'{scale}.svg', bbox_inches='tight',
                                  transparent=True)
                plt.close(fig=fig)

                # save correlations b/w surrogates
                corrs = nnstats.efficient_pearsonr(*surrs)[0]
                fname = OUTDIR / name / method / f'{scale}.csv'
                fname.parent.mkdir(exist_ok=True, parents=True)
                np.savetxt(fname, corrs, fmt='%.10f')

            # make distplot of the correlations b/w surrogates for methods
            fig, ax = plt.subplots(1, 1)
            for method in METHODS:
                corrs = np.loadtxt(OUTDIR / name / method / f'{scale}.csv')
                ax = sns.kdeplot(corrs, label=method, shade=True, ax=ax)
            sns.despine(ax=ax, left=True)
            ax.set(xlim=(0, ax.get_xlim()[1]), xticks=[0, 0.5, 1], yticks=[])
            fname = FIGDIR / name / 'correlations' / f'{scale}.svg'
            fname.parent.mkdir(exist_ok=True)
            fig.savefig(fname, transparent=True, bbox_inches='tight')
            plt.close(fig=fig)

    burt2018, burt2020, moran = [], [], []
    for method, corrs in zip(METHODS, [burt2018, burt2020, moran]):
        for name, annotations in parcellations.items():
            for scale, annot in annotations.items():
                corrs.append(
                    np.loadtxt(OUTDIR / name / method / f'{scale}.csv')
                )
        corrs = np.hstack(corrs)
        mean, (lo, hi) = get_distcorr_stats(corrs)
        print(f'{method}: {mean:.4f} [{lo:.4f}--{hi:.4f}]')
