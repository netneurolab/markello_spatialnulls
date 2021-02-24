# -*- coding: utf-8 -*-
"""
Creates surrogate maps for Neurosynth data using Burt 2018 + 2020 methods. In
both cases, surrogate maps are stored as resampling arrays of the original maps
and are saved to `data/derivatives/surrogates/<atlas>/<method>/neurosynth`.
"""

from pathlib import Path

from joblib import Parallel, delayed
import numpy as np

from parspin import surrogates, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
NSDIR = Path('./data/derivatives/neurosynth').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
SURRDIR = Path('./data/derivatives/surrogates').resolve()
SEED = 1234
N_PROC = 36
N_PERM = 10000


def burt2018_surrogates(name, scale):
    """
    Generates surrogates according to Burt et al., 2018, Nat Neuro

    Parameters
    ----------
    atlas : {'atl-cammoun2012', 'atl-schaefer2018'}, str
        Name of atlas for which to load data
    scale : str
        Scale of atlas to use
    """

    # load data + distance matrix for given parcellation
    lh, rh, concepts = surrogates.load_data(NSDIR, name, scale)
    dlh, drh = surrogates.load_dist(DISTDIR, name, scale)

    # boxcox transformation requires positive values; shift
    shift = abs(min(np.min(lh), np.min(rh))) + 0.1
    lh, rh = lh + shift, rh + shift

    outdir = SURRDIR / name / 'burt2018' / 'neurosynth'
    Parallel(n_jobs=N_PROC)(delayed(surrogates.burt2018_surrogates)(
        lh[:, i], rh[:, i], dlh, drh,
        fname=outdir / concepts[i] / f'{scale}_surrogates.csv',
        n_perm=N_PERM
    ) for i in putils.trange(len(concepts), desc=f'Burt 2020 ({scale})'))


def burt2020_surrogates(name, scale):
    """
    Generates surrogates according to Burt et al., 2020, NeuroImage

    Parameters
    ----------
    atlas : {'atl-cammoun2012', 'atl-schaefer2018'}, str
        Name of atlas for which to load data
    scale : str
        Scale of atlas to use
    """

    # load data + distance matrix for given parcellation
    lh, rh, concepts = surrogates.load_data(NSDIR, name, scale)
    dlh, drh = surrogates.load_dist(DISTDIR, name, scale)

    outdir = SURRDIR / name / 'burt2020' / 'neurosynth'
    Parallel(n_jobs=N_PROC)(delayed(surrogates.burt2020_surrogates)(
        lh[:, i], rh[:, i], dlh, drh,
        fname=outdir / concepts[i] / f'{scale}_surrogates.csv',
        n_perm=N_PERM, seed=SEED
    ) for i in putils.trange(len(concepts), desc=f'Burt 2020 ({scale})'))


if __name__ == '__main__':
    # get cammoun + schaefer parcellations
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    for name, annotations in parcellations.items():
        print(f'PARCELLATION: {name}')
        for scale, annot in annotations.items():
            burt2018_surrogates(name, scale)
            burt2020_surrogates(name, scale)
