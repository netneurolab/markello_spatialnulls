# -*- coding: utf-8 -*-

import os
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib
import numpy as np
from scipy import ndimage
import tqdm

from netneurotools import datasets as nndata, plotting as nnplot

PARULA = LinearSegmentedColormap.from_list('parula', [
    [0.2081000000, 0.1663000000, 0.5292000000],
    [0.2116238095, 0.1897809524, 0.5776761905],
    [0.2122523810, 0.2137714286, 0.6269714286],
    [0.2081000000, 0.2386000000, 0.6770857143],
    [0.1959047619, 0.2644571429, 0.7279000000],
    [0.1707285714, 0.2919380952, 0.7792476190],
    [0.1252714286, 0.3242428571, 0.8302714286],
    [0.0591333333, 0.3598333333, 0.8683333333],
    [0.0116952381, 0.3875095238, 0.8819571429],
    [0.0059571429, 0.4086142857, 0.8828428571],
    [0.0165142857, 0.4266000000, 0.8786333333],
    [0.0328523810, 0.4430428571, 0.8719571429],
    [0.0498142857, 0.4585714286, 0.8640571429],
    [0.0629333333, 0.4736904762, 0.8554380952],
    [0.0722666667, 0.4886666667, 0.8467000000],
    [0.0779428571, 0.5039857143, 0.8383714286],
    [0.0793476190, 0.5200238095, 0.8311809524],
    [0.0749428571, 0.5375428571, 0.8262714286],
    [0.0640571429, 0.5569857143, 0.8239571429],
    [0.0487714286, 0.5772238095, 0.8228285714],
    [0.0343428571, 0.5965809524, 0.8198523810],
    [0.0265000000, 0.6137000000, 0.8135000000],
    [0.0238904762, 0.6286619048, 0.8037619048],
    [0.0230904762, 0.6417857143, 0.7912666667],
    [0.0227714286, 0.6534857143, 0.7767571429],
    [0.0266619048, 0.6641952381, 0.7607190476],
    [0.0383714286, 0.6742714286, 0.7435523810],
    [0.0589714286, 0.6837571429, 0.7253857143],
    [0.0843000000, 0.6928333333, 0.7061666667],
    [0.1132952381, 0.7015000000, 0.6858571429],
    [0.1452714286, 0.7097571429, 0.6646285714],
    [0.1801333333, 0.7176571429, 0.6424333333],
    [0.2178285714, 0.7250428571, 0.6192619048],
    [0.2586428571, 0.7317142857, 0.5954285714],
    [0.3021714286, 0.7376047619, 0.5711857143],
    [0.3481666667, 0.7424333333, 0.5472666667],
    [0.3952571429, 0.7459000000, 0.5244428571],
    [0.4420095238, 0.7480809524, 0.5033142857],
    [0.4871238095, 0.7490619048, 0.4839761905],
    [0.5300285714, 0.7491142857, 0.4661142857],
    [0.5708571429, 0.7485190476, 0.4493904762],
    [0.6098523810, 0.7473142857, 0.4336857143],
    [0.6473000000, 0.7456000000, 0.4188000000],
    [0.6834190476, 0.7434761905, 0.4044333333],
    [0.7184095238, 0.7411333333, 0.3904761905],
    [0.7524857143, 0.7384000000, 0.3768142857],
    [0.7858428571, 0.7355666667, 0.3632714286],
    [0.8185047619, 0.7327333333, 0.3497904762],
    [0.8506571429, 0.7299000000, 0.3360285714],
    [0.8824333333, 0.7274333333, 0.3217000000],
    [0.9139333333, 0.7257857143, 0.3062761905],
    [0.9449571429, 0.7261142857, 0.2886428571],
    [0.9738952381, 0.7313952381, 0.2666476190],
    [0.9937714286, 0.7454571429, 0.2403476190],
    [0.9990428571, 0.7653142857, 0.2164142857],
    [0.9955333333, 0.7860571429, 0.1966523810],
    [0.9880000000, 0.8066000000, 0.1793666667],
    [0.9788571429, 0.8271428571, 0.1633142857],
    [0.9697000000, 0.8481380952, 0.1474523810],
    [0.9625857143, 0.8705142857, 0.1309000000],
    [0.9588714286, 0.8949000000, 0.1132428571],
    [0.9598238095, 0.9218333333, 0.0948380952],
    [0.9661000000, 0.9514428571, 0.0755333333],
    [0.9763000000, 0.9831000000, 0.0538000000],
])

DROP = [  # regions that should always be dropped from analyses
    'lh_unknown', 'rh_unknown',
    'lh_corpuscallosum', 'rh_corpuscallosum',
    'lh_Background+FreeSurfer_Defined_Medial_Wall',
    'rh_Background+FreeSurfer_Defined_Medial_Wall',
]


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
        return
    np.savetxt(fname, data, delimiter=',', fmt=fmt)


def parcellate(mgh, annot):
    """
    Parcellates surface `mgh` file with `annot`

    Parameters
    ----------
    mgh : str or os.PathLike
        Surface data to be parcellated
    annot : str or os.PathLike
        FreeSurfer-style annotation file matching `mgh`

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

    return ndimage.mean(np.squeeze(img), labels, np.unique(labels))


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

    for hemi in ('lh', 'rh'):
        # load relevant distance matrix
        fn = pathify(dist_dir) / atlas / medial / f'{scale}_{hemi}_dist.csv'
        dist = np.loadtxt(fn, delimiter=',')

        if inverse:
            np.fill_diagonal(dist, 1)
            dist **= -1

        # get indices of data that correspond to relevant `hemi` and subset
        idx = [n for n, f in enumerate(data.index)if f.startswith(hemi)]
        hemidata = np.squeeze(np.asarray(data.iloc[idx]))

        yield hemidata, dist, idx


def save_brainmap(data, lh, rh, fname, **kwargs):
    """
    Plots parcellated `data` to the surface and saves to `fname`

    Parameters
    ----------
    plot : array_like
        Parcellated data to be plotted to the surface. Should be in the order
        {left,right} hemisphere
    {lh,rh} : str or os.pathLike
        Annotation files for the {left,right} hemisphere, matching `data`.
        By default assumes these are 'fsaverage' resolution. Set `subject_id`
        kwarg if different.
    fname : str or os.PathLike
        Filepath where plotted figure should be saved
    """

    opts = dict(
        alpha=1.0, views=['lat'], colormap=PARULA, colorbar=True,
        surf='inflated', subject_id='fsaverage', size_per_view=500,
        offscreen=True, noplot=[b'unknown', b'corpuscallosum',
                                b'Background+FreeSurfer_Defined_Medial_Wall']
    )
    opts.update(kwargs)
    fig = nnplot.plot_fsaverage(data, lhannot=lh, rhannot=rh, **opts)
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.save_image(fname)
    fig.close()


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
