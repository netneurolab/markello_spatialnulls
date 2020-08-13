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
from parspin import burt, utils as putils

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Verdana']
plt.rcParams['font.size'] = 28.0

SEED = 1234
N_SURROGATES = 10000
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


if __name__ == "__main__":
    FIGDIR.mkdir(exist_ok=True, parents=True)
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    for name, annotations in parcellations.items():
        print(f'PARCELLATION: {name}')
        for scale, annot in annotations.items():
            print(f'Comparing surrogates for {scale}')

            # load T1w/T2w for given parcellation + resolution
            data = pd.read_csv(HCPDIR / name / f'{scale}.csv', index_col=0)
            data = data.drop([l for l in data.index if l in putils.DROP])
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
                            surr = np.row_stack([
                                burt.make_surrogate(dist, hd, seed=n)
                                for n in range(N_SURROGATES)
                            ])
                        elif method == 'burt2020':
                            base = Base(hd, dist, resample=True, seed=SEED)
                            surr = base(N_SURROGATES)
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
                for n, medial in enumerate(['medial', 'nomedial']):
                    vmin, vmax = np.percentile(surrs, [2.5, 97.5])
                    fname = figdir / medial / f'{scale}.png'
                    fname.parent.mkdir(exist_ok=True)
                    putils.save_brainmap(surrs[n][:, 0], annot.lh, annot.rh,
                                         fname, colormap='coolwarm',
                                         colorbar=False, vmin=vmin, vmax=vmax,
                                         subject_id='fsaverage5')

                # scatter plot of example brains
                fig, ax = plt.subplots(1, 1)
                ax.scatter(surrs[0][:, 0], surrs[1][:, 0], edgecolor='white',
                           s=75, facecolor=np.array([112, 146, 255]) / 255)
                for side in ['right', 'top']:
                    ax.spines[side].set_visible(False)
                ax.set(xlabel='with medial wall', ylabel='without medial wall')
                l, h = ax.get_xlim()
                ax.plot([l, h], [l, h], c='gray')
                ax.figure.savefig(figdir / f'{scale}.svg', bbox_inches='tight',
                                  transparent=True)
                plt.close(fig=ax.figure)

                # save correlations b/w surrogates
                corrs = nnstats.efficient_pearsonr(*surrs)[0]
                fname = OUTDIR / name / method / f'{scale}.csv'
                fname.parent.mkdir(exist_ok=True, parents=True)
                np.savetxt(fname, corrs, fmt='%.10f')

            # make distplot of the correlations b/w surrogates for methods
            fig, ax = plt.subplots(1, 1)
            for method in METHODS:
                corrs = np.loadtxt(OUTDIR / name / method / f'{scale}.csv')
                sns.distplot(corrs, label=method)
            sns.despine(ax=ax, left=True)
            ax.set(xlim=(0, ax.get_xlim()[1]), xticks=[0, 0.5, 1], yticks=[])
            fname = FIGDIR / name / 'correlations' / f'{scale}.svg'
            fname.parent.mkdir(exist_ok=True)
            fig.savefig(fname, transparent=True, bbox_inches='tight')
            plt.close(fig=fig)
