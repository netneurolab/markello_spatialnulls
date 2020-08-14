# -*- coding: utf-8 -*-
"""
Helper code for generating surrogate resampling matrices for Burt 2018 and Burt
2020 methods
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from parspin import burt, utils as putils

from brainsmash import mapgen

warnings.simplefilter('ignore', category=np.VisibleDeprecationWarning)


def load_data(data_dir, atlas, scale):
    """
    Loads parcellated data for given `atlas` and `scale` from `data_dir`

    Parameters
    ----------
    data_dir : str or os.PathLike
        Directory where data are stored
    atlas : {'atl-cammoun2012', 'atl-schaefer2018'}, str
        Name of atlas for which to load data
    scale : str
        Scale of atlas to use

    Returns
    -------
    lh : (N, T) numpy.ndarray
        Left hemisphere data, where `N` is regions and `T` is features
    rh : (M, T) numpy.ndarray
        Right hemisphere data, where `M` is regions and `T` is features
    labels : numpy.ndarray
        Feature labels corresponding to columns of `{lh,rh}_data`
    """

    data_dir = Path(data_dir)
    data = pd.read_csv(data_dir / atlas / f'{scale}.csv', index_col=0)

    # drop medial stuff
    todrop = np.array(putils.DROP)[np.isin(putils.DROP, data.index)]
    if len(todrop) > 0:
        data = data.drop(todrop, axis=0)

    # get indices of diff hemispheres
    idx_lh = [n for n, f in enumerate(data.index) if 'lh_' in f]
    idx_rh = [n for n, f in enumerate(data.index) if 'rh_' in f]

    # get data array
    labels = np.asarray(data.columns)
    data = np.asarray(data).squeeze()

    return data[idx_lh], data[idx_rh], labels


def load_dist(dist_dir, atlas, scale):
    """
    Loads parcellated distance matrix for given `atlas` and `scale`

    Parameters
    ----------
    dist_dir : str or os.PathLike
        Directory where distance matrices are stored
    atlas : {'atl-cammoun2012', 'atl-schaefer2018'}, str
        Name of atlas for which to load data
    scale : str
        Scale of atlas to use

    Returns
    -------
    lh : (N, N) numpy.ndarray
        Left hemisphere distance matrix, where `N` is regions
    rh : (M, M) numpy.ndarray
        Right hemisphere distance matrix, where `M` is regions
    """

    dist_dir = Path(dist_dir)

    dist = ()
    for hemi in ('lh', 'rh'):
        fn = dist_dir / atlas / 'nomedial' / f'{scale}_{hemi}_dist.csv'
        dist += (np.loadtxt(fn, delimiter=','),)
    return dist


def inverse_argsort(arr):
    """ Small helper function to generate inverse argsort of `arr`

    Parameters
    ----------
    arr : array_like
        Input data array

    Returns
    -------
    inverse : np.ndarray
        Inverse argsort of `arr`

    Notes
    -----
    All credit to: https://stackoverflow.com/a/46773478
    """

    forward = np.argsort(arr)
    inverse = np.empty_like(forward)
    inverse[forward] = np.arange(len(forward))
    return inverse


def burt2018_surrogates(lh, rh, dlh, drh, fname, n_perm=10000):
    """
    Generates surrogates using Burt et al., 2018, Nat Neuro method

    Parameters
    ----------
    lh, rh : (N,) array_like
        Input feature data for left and right hemisphers
    dlh, drh : (N, N) array_like
        Distance matrices for left and right hemispheres
    fname : str or os.PathLike
        Path to where generated surrogate resampling array should be saved
    n_perm : int, optional
        Number of surrogates to generate. Default: 10000
    """

    if fname.exists():
        return fname

    data = np.hstack((lh, rh))
    lhord, rhord = np.argsort(lh), np.argsort(rh)

    # we've gotta generate surrogates separately for right + left hemispheres
    # because our rho / d0 estimation depends on the geodesic distance matrix
    # and we can't get that for inter-hemispheric connections
    lh_estimates = burt.estimate_rho_d0(dlh, lh)
    rh_estimates = burt.estimate_rho_d0(drh, rh)

    surrogates = np.zeros((len(lh) + len(rh), n_perm), dtype='int32')
    for n in range(n_perm):
        # generate the surrogates separately and then stack them together. pass
        # the same seed for both hemispheres to try and keep things consistent
        ls, lo = burt.make_surrogate(dlh, lh, *lh_estimates,
                                     seed=n, return_order=True)
        rs, ro = burt.make_surrogate(drh, rh, *rh_estimates,
                                     seed=n, return_order=True)
        surrogates[:, n] = np.hstack((lhord[np.argsort(lo)],
                                      rhord[np.argsort(ro)] + len(lh)))

        # we're only saving the ORDER of the surrogates. surrogate maps
        # (equivalent to stacking outputs `ls` + `rs`) can be regenerated with:
        #
        #     >>> surr = data[surrogates[:, n]]
        #
        # confirm that this works as expected:
        assert np.allclose(data[surrogates[:, n]], np.hstack((ls, rs)))

    putils.save_dir(fname, surrogates)

    return fname


def burt2020_surrogates(lh, rh, dlh, drh, fname, n_perm=10000, seed=None):
    """
    Generates surrogates using Burt et al., 2020, NeuroImage method

    Parameters
    ----------
    lh, rh : (N,) array_like
        Input feature data for left and right hemisphers
    dlh, drh : (N, N) array_like
        Distance matrices for left and right hemispheres
    fname : str or os.PathLike
        Path to where generated surrogate resampling array should be saved
    n_perm : int, optional
        Number of surrogates to generate. Default: 10000
    seed : {NoneType, int, RandomState instance}, optional
        Random seed for reproducibility in generated surrogates
    """

    if fname.exists():
        return fname

    # generate surrogates
    # we need to pass `resample=True` otherwise we get VASTLY different values
    # for the hemispheres which is...problematic
    lhbase = mapgen.Base(lh, dlh, resample=True, seed=seed)
    rhbase = mapgen.Base(rh, drh, resample=True, seed=seed)
    ls, rs = lhbase(n_perm), rhbase(n_perm)

    # get re-ordering indices (so we can just save those)
    data = np.hstack((lh, rh))
    lhord, rhord = np.argsort(lh), np.argsort(rh)
    surrogates = np.zeros((len(lh) + len(rh), n_perm), dtype='int32')
    for n in range(n_perm):
        lo, ro = lhord[inverse_argsort(ls[n])], rhord[inverse_argsort(rs[n])]
        surrogates[:, n] = np.hstack((lo, ro + len(lh)))

        # check that we get what we expect with the transform
        assert np.allclose(data[surrogates[:, n]], np.hstack((ls[n], rs[n])))

    putils.save_dir(fname, surrogates)

    return fname
