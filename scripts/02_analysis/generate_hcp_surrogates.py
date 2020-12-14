# -*- coding: utf-8 -*-
"""
Creates surrogate maps for HCP myelin data using Burt 2018 + 2020 methods. In
both cases, surrogate maps are stored as resampling arrays of the original maps
and are saved to `data/derivatives/surrogates/<atlas>/<method>/hcp`.
"""

from pathlib import Path

from joblib import Parallel, delayed

from parspin import surrogates, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
HCPDIR = Path('./data/derivatives/hcp').resolve()
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

    fn = SURRDIR / name / 'burt2018' / 'hcp' / f'{scale}_surrogates.csv'
    if fn.exists():
        return

    # load data + distance matrix for given parcellation
    lh, rh = surrogates.load_data(HCPDIR, name, scale)[:-1]
    dlh, drh = surrogates.load_dist(DISTDIR, name, scale)

    # generate surrogates and save to disk
    surrogates.burt2018_surrogates(lh, rh, dlh, drh, fname=fn, n_perm=N_PERM)


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

    fn = SURRDIR / name / 'burt2020' / 'hcp' / f'{scale}_surrogates.csv'
    if fn.exists():
        return

    # load data + distance matrix for given parcellation
    lh, rh = surrogates.load_data(HCPDIR, name, scale)[:-1]
    dlh, drh = surrogates.load_dist(DISTDIR, name, scale)

    # generate surrogates and save to disk
    surrogates.burt2020_surrogates(lh, rh, dlh, drh, fname=fn,
                                   n_perm=N_PERM, seed=SEED)


if __name__ == '__main__':
    # get cammoun + schaefer parcellations
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    for func in (burt2018_surrogates, burt2020_surrogates):
        Parallel(n_jobs=N_PROC)(
            delayed(func)(name, scale)
            for (name, annotations) in parcellations.items()
            for (scale, annot) in annotations.items()
        )
