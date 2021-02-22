# -*- coding: utf-8 -*-

import os
from pathlib import Path
import warnings

import nibabel as nib
import numpy as np
from scipy import ndimage
import seaborn as sns
import tqdm

from netneurotools import datasets as nndata
from netneurotools.freesurfer import _decode_list

DROP = [  # regions that should always be dropped from analyses
    'lh_unknown', 'rh_unknown',
    'lh_corpuscallosum', 'rh_corpuscallosum',
    'lh_Background+FreeSurfer_Defined_Medial_Wall',
    'rh_Background+FreeSurfer_Defined_Medial_Wall',
]


def rgb255(x):
    return np.around(np.asarray(x) * 255)


REDS = rgb255(sns.color_palette('Reds', 7, desat=0.7)[-5:])
BLUES = rgb255(sns.color_palette('Blues', 4, desat=0.5)[-3:])
PURPLES = rgb255(sns.color_palette('Purples', 5, desat=0.8))[[2, 4]]
SPATHUES = list(np.r_[PURPLES, REDS, BLUES] / 255)
PARCHUES = list(np.array([[26, 146, 0], [222, 57, 90], [131, 131, 131]]) / 255)


def pathify(path):
    """
    Convenience function for coercing a potential pathlike to a Path object

    Parameter
    ---------
    path : str or os.PathLike
        Path to be checked for coercion to pathlib.Path object

    Returns
    -------
    path : pathlib.Path
    """

    if isinstance(path, (str, os.PathLike)):
        path = Path(path)
    return path


def get_cammoun_schaefer(vers='fsaverage5', data_dir=None, networks='7'):
    """
    Returns Cammoun 2012 and Schaefer 2018 atlases as dictionary

    Parameters
    ----------
    vers : str, optional
        Which version of the atlases to get. Default: 'fsaverage5'
    data_dir : str or os.PathLike, optional
        Data directory where downloaded atlases should be stored. If not
        specified will default to $NNT_DATA or ~/nnt-data
    networks : {'7', '17'}, optional
        Which networks to get for Schaefer 2018 atlas. Default: '7'

    Returns
    -------
    atlases : dict
        Where keys are 'atl-cammoun2012' and 'atl-schaefer2018'
    """
    cammoun = nndata.fetch_cammoun2012(vers, data_dir=data_dir)
    schaefer = nndata.fetch_schaefer2018('fsaverage5', data_dir=data_dir)
    schaefer = {k: schaefer.get(k) for k in schaefer.keys()
                if 'Parcels7Networks' in k}

    return {'atl-cammoun2012': cammoun, 'atl-schaefer2018': schaefer}


def save_dir(fname, data, overwrite=True):
    """
    Saves `data` to `fname`, creating any necessary intermediate directories

    Parameters
    ----------
    fname : str or os.PathLike
        Output filename for `data`
    data : array_like
        Data to be saved to disk
    """

    fname = Path(fname).resolve()
    fname.parent.mkdir(parents=True, exist_ok=True)
    fmt = '%.10f' if data.dtype.kind == 'f' else '%d'
    if fname.exists() and not overwrite:
        warnings.warn(f'{fname} already exists; not overwriting')
        return
    np.savetxt(fname, data, delimiter=',', fmt=fmt)


def parcellate(mgh, annot, drop=None):
    """
    Parcellates surface `mgh` file with `annot`

    Parameters
    ----------
    mgh : str or os.PathLike
        Surface data to be parcellated
    annot : str or os.PathLike
        FreeSurfer-style annotation file matching `mgh`
    drop : list-of-str, optional
        List of parcels to `drop` from parcellated data. Default: None

    Returns
    -------
    data : numpy.ndarray
        Parcellated data from `mgh`
    """

    try:
        img = nib.load(mgh).dataobj
    except ValueError:
        img = mgh

    labels, ctab, names = nib.freesurfer.read_annot(annot)
    data = ndimage.mean(np.squeeze(img), labels, np.unique(labels))

    if drop is not None:
        if isinstance(drop, str):
            drop = [drop]
        drop, names = _decode_list(drop), _decode_list(names)
        drop = np.intersect1d(names, drop)
        data = np.delete(data, [names.index(f) for f in drop])

    return data


def get_names(*, lh, rh):
    """
    Gets parcel labels in `{l,r}h` files and prepends hemisphere designation

    Parameters
    ----------
    {l,r}h : str or os.PathLike
        FreeSurfer-style annotation files for the left / right hemisphere

    Returns
    -------
    names : list
        List of parcel names with 'lh_' or 'rh_' prepended to names
    """

    names = []
    for fn, hemi in ((lh, 'lh'), (rh, 'rh')):
        n = nib.freesurfer.read_annot(fn)[-1]
        names += [
            f'{hemi}_{fn.decode()}' if hasattr(fn, 'decode') else fn
            for fn in n
        ]
    return names


def trange(n_iter, verbose=True, **kwargs):
    """
    Wrapper for :obj:`tqdm.trange` with some default options set

    Parameters
    ----------
    n_iter : int
        Number of iterations for progress bar
    verbose : bool, optional
        Whether to return an :obj:`tqdm.tqdm` progress bar instead of a range
        generator. Default: True
    kwargs
        Key-value arguments provided to :func:`tqdm.trange`

    Returns
    -------
    progbar : :obj:`tqdm.tqdm`
    """

    form = ('{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
            ' | {elapsed}<{remaining}')
    defaults = dict(ascii=True, leave=False, bar_format=form)
    defaults.update(kwargs)

    return tqdm.trange(n_iter, disable=not verbose, **defaults)


def yield_data_dist(dist_dir, atlas, scale, data, medial=False, inverse=True):
    """
    Yields hemisphere-specific data and distance matrices

    Parameters
    ----------
    dist_dir : str or os.PathLike
        Path to directory where distance matrices are stored
    atlas : {'atl-cammoun2012', 'atl-schaefer2018'}, str
        Name of atlas for which to load data
    scale : str
        Scale of atlas to use
    data : pandas.DataFrame
        Dataframe where index has labels indicating to which hemisphere each
        value belongs
    medial : bool, optional
        Whether to load distance matrix with medial wall travel. Default: False
    inverse : bool, optional
        Whether to return inverse distance matrix. Default: True

    Yields
    -------
    hemidata : (N,) numpy.ndarray
        Data for {left, right} hemisphere
    dist : (N, N) numpy.ndarray
        Distance matrix for {left, right} hemisphere
    index : (N,) list
        Indices of provided `data` that yielded `hemidata`
    """

    medial = ['nomedial', 'medial'][medial]

    for n, hemi in enumerate(('lh', 'rh')):
        # load relevant distance matrix
        fn = pathify(dist_dir) / atlas / medial / f'{scale}_{hemi}_dist.csv'
        npy = fn.with_suffix('.npy')
        if npy.exists():
            dist = np.load(npy, allow_pickle=False, mmap_mode='c')
        else:
            dist = np.loadtxt(fn, delimiter=',')
            np.save(npy, dist, allow_pickle=False)

        if inverse:
            np.fill_diagonal(dist, 1)
            dist **= -1

        # get indices of data that correspond to relevant `hemi` and subset.
        # if data is not a pandas DataFrame with hemisphere information assume
        # we can split the data equally in 2
        try:
            idx = [n for n, f in enumerate(data.index)if f.startswith(hemi)]
            hemidata = np.squeeze(np.asarray(data.iloc[idx]))
        except AttributeError:
            idx = np.arange(n * (len(data) // 2), (n + 1) * (len(data) // 2))
            hemidata = np.squeeze(data[idx])

        yield hemidata, dist, np.asarray(idx)


def drop_unknown(data, drop=DROP):
    """
    Removes rows from `data` corresponding to entries in `drop`

    If there is no overlap between `data.index` and `drop` then nothing is done

    Parameters
    ----------
    data : pandas.DataFrame
        Data from which `drop` should be removed
    drop : array_like
        Indices or index names of rows that should be removed from `data`. If
        not supplied then `parspin.utils.DROP` is used. Default: None

    Returns
    -------
    data : pandas.DataFrame
        Provided dataframe with `drop` rows removed
    """

    todrop = np.array(drop)[np.isin(drop, data.index)]
    if len(todrop) > 0:
        data = data.drop(todrop, axis=0)

    return data
