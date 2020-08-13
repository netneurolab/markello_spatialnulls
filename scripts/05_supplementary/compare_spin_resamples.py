# -*- coding: utf-8 -*-
"""
Compares spin resampling arrays when using different parcel centroid definition
methods and different nulls that rely on parcel centroids
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import seaborn as sns

from netneurotools import freesurfer as nnsurf, stats as nnstats
from parspin import utils as putils

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Verdana']
plt.rcParams['font.size'] = 28.0

ROIDIR = Path('./data/raw/rois').resolve()
FIGDIR = Path('./figures/supplementary/comp_spins/').resolve()


def generate_dist(annot):
    """
    Parameters
    ----------
    annot : (2,) namedtuple
        With entries ('lh', 'rh') corresponding to annotation files for
        specified hemisphere for `fsaverage5` resolution

    Returns
    -------
    dist : (90, 90) np.ndarray
        Distance matrix for resamplings from different null frameworks and
        parcel centroid definition methods
    """

    # generate spins for all the parcel-based rotation methods using
    # three different methods for generating parcel centroid estimates
    orig_spins, vasa_spins, hung_spins = [], [], []
    for sptype, spins in zip(['original', 'vasa', 'hungarian'],
                             [orig_spins, vasa_spins, hung_spins]):
        for method in ['average', 'surface', 'geodesic']:
            # get parcel centroids using specified centroid method
            coords, hemi = nnsurf.find_parcel_centroids(
                lhannot=annot.lh, rhannot=annot.rh,
                version='fsaverage5', method=method
            )
            # generate 10 spin resamples for specified spin method
            out = nnstats.gen_spinsamples(coords, hemi, n_rotate=10,
                                          method=sptype, seed=1234)
            spins.append(out)

    # stack and transpose for easy distance calculation
    all_spins = np.row_stack((
        np.column_stack(orig_spins).T,
        np.column_stack(vasa_spins).T,
        np.column_stack(hung_spins).T
    ))
    all_dist = cdist(all_spins, all_spins, metric='hamming')

    return all_dist


def plot_full_heatmap(dist, fname):
    """
    Plots full 90 x 90 `dist` matrix and saves to `fname`

    Parameters
    ----------
    dist : (90, 90) array_like
        Full distance matrix for resampling arrays
    fname : str or os.PathLike
        Path to where generated figure should be saved
    """

    # plot the matrix
    fig, ax = plt.subplots(1, 1)
    ax = sns.heatmap(all_dist, vmin=0, vmax=1, cmap='YlGnBu_r', ax=ax)
    labels = ['vazquez-\nrodriguez', 'vasa', 'hungarian']

    # add some rectangles here to delineate the different centroid
    # calculation methods + null models
    # black = null model (vazquez-rodriguez, vasa, hungarian)
    # gray = centroid calculation method (surface, average, geodesic)
    for m in np.arange(10, 90, 10):
        if m in [30, 60]:
            ax.hlines(m, *ax.get_xlim(), linewidth=1, color='k')
            ax.vlines(m, *ax.get_ylim(), linewidth=1, color='k')
        ax.hlines(m, *ax.get_xlim(), linewidth=0.5, color='gray')
        ax.vlines(m, *ax.get_ylim(), linewidth=0.5, color='gray')

    # clean up labels + add colorbar
    ax.set(ylim=(90, 0), xticks=[15, 45, 75], yticks=[15, 45, 75])
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)
    ax.tick_params(length=0)

    # save to disk
    fig.savefig(fname, bbox_inches='tight', transparent=True)
    plt.close(fig=fig)


def plot_reduced_heatmap(dist, fname):
    """
    Plots reduced 9 x 9 `dist` matrrix and saves to `fname`

    Parameters
    ----------
    dist : (9, 9) array_like
        Reduced (i.e., averaged) distance matrix for resampling arrays
    fname : str or os.PathLike
        Path to where generated figure should be saved
    """

    fig, ax = plt.subplots(1, 1)
    ax = sns.heatmap(avgs, vmin=0, vmax=1, cmap='YlGnBu_r', ax=ax)
    labels = ['vazquez-\nrodriguez', 'vasa', 'hungarian']

    # clean up and make the black outlines delineating the diff null models
    ax.set(ylim=(9, 0), xticks=[1.5, 4.5, 7.5], yticks=[1.5, 4.5, 7.5])
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)
    ax.vlines([3, 6], *ax.get_xlim(), linewidth=1, color='k')
    ax.hlines([3, 6], *ax.get_ylim(), linewidth=1, color='k')
    ax.tick_params(length=0)

    fig.savefig(fname, bbox_inches='tight', transparent=True)
    plt.close(fig=fig)


if __name__ == "__main__":
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    for name, annotations in parcellations.items():
        print(f'PARCELLATION: {name}')
        (FIGDIR / name).mkdir(exist_ok=True, parents=True)
        for scale, annot in annotations.items():
            print(f'Comparing spins for {scale}')
            all_dist = generate_dist(annot)

            # plot the full matrix (we likely won't use this but just in case)
            fname = FIGDIR / name / f'{scale}_full.svg'
            plot_full_heatmap(all_dist, fname)

            # now, average the diagonals of all the sections
            avgs = np.zeros((9, 9))
            for n, r in enumerate(range(0, 90, 10)):
                for m, c in enumerate(range(0, 90, 10)):
                    avgs[n, m] = np.mean(np.diag(all_dist[r:r + 10, c:c + 10]))

            # aaand, plot those averaged diagonals as a "reduced" heatmap
            # this is _much_ more interpretable and easy-to-digest than the
            # the previous one, so we'll likely use that
            fname = FIGDIR / name / f'{scale}_reduced.svg'
            plot_reduced_heatmap(avgs, fname)
