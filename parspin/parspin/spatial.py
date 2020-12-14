# -*- coding: utf-8 -*-
"""
Functions for calculating and manipulating spatial autocorrelation
"""

import os
import tempfile

import gstools as gs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import meshio
import nibabel as nib
from nilearn.plotting import plot_surf_stat_map
import numpy as np
from scipy import fftpack, stats as sstats

from netneurotools.datasets import fetch_fsaverage, make_correlated_xy
from netneurotools.freesurfer import check_fs_subjid
from netneurotools.utils import run

VOL2SURF = 'mri_vol2surf --src {} --out {} --hemi {} --mni152reg ' \
           '--trgsubject fsaverage5 --projfrac 0.5 --interp nearest'
MSEED = 4294967295


def morans_i(dist, y, normalize=False, local=False):
    """
    Calculates Moran's I from distance matrix `dist` and brain map `y`

    Parameters
    ----------
    dist : (N, N) array_like
        Distance matrix between `N` regions / vertices / voxels / whatever
    y : (N,) array_like
        Brain map variable of interest
    normalize : bool, optional
        Whether to normalize rows of distance matrix prior to calculation.
        Default: False
    local : bool, optional
        Whether to calculate local Moran's I instead of global. Default: False

    Returns
    -------
    i : float
        Moran's I, measure of spatial autocorrelation
    """

    # convert distance matrix to weights
    with np.errstate(divide='ignore'):
        dist = 1 / dist
    np.fill_diagonal(dist, 0)

    # normalize rows, if desired
    if normalize:
        dist /= dist.sum(axis=-1, keepdims=True)

    # calculate Moran's I
    z = y - y.mean()
    if local:
        with np.errstate(all='ignore'):
            z /= y.std()

    zl = (dist * z).sum(axis=-1)
    den = (z * z).sum()

    if local:
        return (len(y) - 1) * z * zl / den

    return len(y) / dist.sum() * (z * zl).sum() / den


def smooth_surface(y, faces, fwhm=6):
    """
    Smooths brain map `y` defined by triangle `faces` with kernel `fwhm`

    Parameters
    ----------
    y : (N,) array_like
        Brain map variable of interest
    faces : (T, 3) array_like
        Triangular faces comprising spatial mesh of `N` vertices
    fwhm : float, optional
        Approximate FWHM of smoothing kernel (in mesh units). Default: 6

    Returns
    -------
    y : (N,) array_like
        Smoothed input
    """

    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2)), axis=1)

    v = len(y)
    ec = [2] * len(edges)
    y1 = np.bincount(edges[:, 0], ec, v) + np.bincount(edges[:, 1], ec, v)

    for i in range(int(np.ceil((fwhm ** 2) / (2 * np.log(2))))):
        ysum = y[edges[:, 0]] + y[edges[:, 1]]
        y = (np.bincount(edges[:, 0], ysum, v)
             + np.bincount(edges[:, 1], ysum, v)) / y1

    return y


def _fftind(x, y, z):
    """
    Return 3D shifted Fourier coordinates

    Returned coordinates are shifted such that zero-frequency component of the
    square grid with shape (x, y, z) is at the center of the spectrum

    Parameters
    ----------
    x,y,z : int
        size of array

    Returns
    -------
    k_ind : (3, x, y, z) np.ndarray
        shifted Fourier coordinates, where:
            k_ind[0] : k_x components
            k_ind[1] : k_y components
            k_ind[2] : k_z components

    Notes
    -----
    see scipy.fftpack.fftshift
    """

    k_ind = np.mgrid[:x, :y, :z]
    zero = np.array([int((n + 1) / 2) for n in [x, y, z]])
    while zero.ndim < k_ind.ndim:
        zero = np.expand_dims(zero, -1)
    k_ind = fftpack.fftshift(k_ind - zero)

    return k_ind


def gaussian_random_field(x, y, z, noise=None, alpha=3.0, normalize=True,
                          seed=None):
    """
    Generate a Gaussian random field with k-space power law |k|^(-alpha/2).

    Parameters
    ----------
    x,y,z : int
        Grid size of generated field
    noise : (x, y, z) array_like, optional
        Noise array to which gaussian smoothing is added. If not provided an
        array will be created by drawing from the standard normal distribution.
        Default: None
    alpha : float (positive), optional
        Power (exponent) of the power-law distribution. Default: 3.0
    normalize : bool, optional
        Normalize the returned field to unit variance. Default: True
    seed : None, int, default_rng, optional
        Random state to seed GRF. Default: None

    Returns
    -------
    gfield : (x, y, z) np.ndarray
        Realization of gaussian random field
    """

    rs = np.random.default_rng(seed)

    if not alpha:
        return rs.normal(size=(x, y, z))

    assert alpha > 0

    # k-space indices
    k_idx = _fftind(x, y, z)

    # define k-space amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power(np.sum([k ** 2 for k in k_idx], axis=0) + 1e-10,
                         -alpha / 2.0)
    amplitude[0, 0, 0] = 0  # remove zero-freq mean shit

    # generate a complex gaussian random field where phi = phi_1 + i*phi_2
    if noise is None:
        noise = rs.normal(size=(x, y, z))
    elif noise.shape != (x, y, z):
        try:
            noise = noise.reshape(x, y, z)
        except ValueError:
            raise ValueError('Provided noise cannot be reshape to target: '
                             f'({x}, {y}, {z})')

    # transform back to real space
    gfield = np.fft.ifftn(np.fft.fftn(noise) * amplitude).real

    if normalize:
        return (gfield - gfield.mean()) / gfield.std()

    return gfield


def create_gstools_grf(len_scale=20, seed=None, **kwargs):
    """
    Generates GRF on surface (fsaverage5)

    Uses gstools.SRF().mesh() to generate GRF

    Parameters
    ----------
    len_scale : int (positive), optional
        Length scale of Gaussian covariance model. Default: 20
    seed : None, int, default_rng, optional
        Random state to seed GRF. Default: None

    Returns
    -------
    data : (20484,) np.ndarray
        Surface representation of GRF
    """

    rs = np.random.default_rng(seed)

    # make the covariance model with specified len scale + the simulation
    model = gs.Gaussian(len_scale=len_scale)
    srf = gs.SRF(model, seed=rs.integers(MSEED))

    fs = fetch_fsaverage('fsaverage5')['sphere']
    data = np.zeros((20484,))
    for n, hemi in enumerate(('lh', 'rh')):
        points, cells = nib.freesurfer.read_geometry(getattr(fs, hemi))
        mesh = meshio.Mesh(points, [('triangles', cells)])
        sl = slice(len(data) // 2 * n, len(data) // 2 * (n + 1))
        # generate the simulated field on the provided mesh
        data[sl] = srf.mesh(mesh, 'points')

    return data


def make_tmpname(suffix):
    """
    Stupid helper function because :man_shrugging:

    Parameters
    ----------
    suffix : str
        Suffix of created filename

    Returns
    -------
    fn : str
        Temporary filename; user is responsible for deletion
    """

    # I don't want to deal with a bunch of nested tempfile.NameTemporaryFile
    # in the create_surface_grf() function so this is the easiest way to do
    # things that's safe from race conditions :man_shrugging:

    fd, fn = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    return fn


def create_surface_grf(noise=None, alpha=3.0, normalize=True, seed=None):
    """
    Generates GRF on surface (fsaverage5)

    Uses gaussian_random_field() and mri_vol2surf to generate GRF

    Parameters
    ----------
    noise : (x, y, z) array_like, optional
        Noise array to which gaussian smoothing is added. If not provided an
        array will be created by drawing from the standard normal distribution.
        Default: None
    alpha : float (positive), optional
        Power (exponent) of the power-law distribution. Default: 3.0
    normalize : bool, optional
        Normalize the returned field to unit variance. Default: True
    seed : None, int, default_rng, optional
        Random state to seed GRF. Default: None

    Returns
    -------
    data : (20484,) np.ndarray
        Surface representation of GRF
    """

    affine = np.eye(4) * 2
    affine[:, -1] = [-90, -90, -72, 1]

    gfield = gaussian_random_field(91, 109, 91, noise=noise, alpha=alpha,
                                   normalize=normalize, seed=seed)
    fn = make_tmpname(suffix='.nii.gz')
    nib.save(nib.nifti1.Nifti1Image(gfield, affine), fn)

    data = np.zeros((20484,))
    for n, hemi in enumerate(('lh', 'rh')):
        outname = make_tmpname(suffix='.mgh')
        run(VOL2SURF.format(fn, outname, hemi), quiet=True)
        sl = slice(len(data) // 2 * n, len(data) // 2 * (n + 1))
        data[sl] = nib.load(outname).get_fdata().squeeze()
        os.remove(outname)

    os.remove(fn)

    return data


def make_surf_plot(data, surf='inflated', version='fsaverage5', **kwargs):
    """
    Generates 2 x 2 surface plot of `data`

    Parameters
    ----------
    data : array_like
        Data to be plotted; should be left hemisphere first
    surf : {'orig', 'white', 'smoothwm', 'pial', 'inflated', 'sphere'}
        Which surface plot should be displaye don
    version : {'fsaverage', 'fsaverage5', 'fsaverage6'}
        Which fsaverage surface to use for plots. Dimensionality of `data`
        should match resolution of surface

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """

    # get plotting kwargs
    opts = dict(colorbar=False, vmax=data.max())
    opts.update(kwargs)
    for key in ['hemi', 'view', 'axes']:
        if key in opts:
            del opts[key]

    data = np.split(data, 2)
    fs = fetch_fsaverage(version)[surf]

    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    for row, view in zip(axes, ['lateral', 'medial']):
        for n, (col, hemi) in enumerate(zip(row, ['lh', 'rh'])):
            fn = getattr(fs, hemi)
            hemi = 'left' if hemi == 'lh' else 'right'

            plot_surf_stat_map(fn, data[n], hemi=hemi, view=view, axes=col,
                               **opts)

    fig.tight_layout()

    return fig


def _mod_medial(data, remove=True):
    """
    Removes (inserts) medial wall from (into) `data` from fsaverage5 surface

    Parameters
    ----------
    data : (20484,) array_like
        Surface data
    remove : bool, optional
        Whether to remove medial wall instead of inserting it. Assumes input
        has (does not have) medial wall. Default: True

    Returns
    -------
    out : np.ndarray
        Provided surface `data` with medial wall removed/inserted
    """

    subj, path = check_fs_subjid('fsaverage5')
    lh, rh = [
        nib.freesurfer.read_label(
            os.path.join(path, subj, 'label', f'{h}.Medial_wall.label')
        )
        for h in ('lh', 'rh')
    ]
    lhm, rhm = np.ones(10242, dtype=bool), np.ones(10242, dtype=bool)
    lhm[lh], rhm[rh] = False, False

    if remove:
        x, y = np.split(data, 2)
        return np.hstack((x[lhm], y[rhm]))
    else:
        x, y = np.split(data, [np.sum(lhm)])
        xd, yd = np.zeros(10242), np.zeros(10242)
        xd[lhm], yd[rhm] = x, y
        return np.hstack((xd, yd))


def matching_multinorm_grfs(corr, tol=0.005, *, alpha=3.0, normalize=True,
                            seed=None, debug=False):
    """
    Generates two surface GRFs (fsaverage5) that correlate at r = `corr`

    Starts by generating two random variables from a multivariate normal
    distribution with correlation `corr`, adds spatial autocorrelation with
    specified `alpha`, and projects to the surface. Continues this procedure
    until two variables are generated that have correlation `corr` on the
    surface.

    Parameters
    ----------
    corr : float
        Desired correlation of generated GRFs
    tol : float
        Tolerance for correlation between generated GRFs
    alpha : float (positive), optional
        Exponent of the power-law distribution. Only used if `use_gstools` is
        set to False. Default: 3.0
    normalize : bool, optional
        Whether to normalize the returned field to unit variance. Default: True
    seed : None, int, default_rng, optional
        Random state to seed GRF generation. Default: None
    debug : bool, optional
        Whether to print debug info

    Return
    ------
    x, y : (20484,) np.ndarray
        Generated surface GRFs
    """

    rs = np.random.default_rng(seed)

    acorr, n = np.inf, 0
    while np.abs(np.abs(acorr) - corr) > tol:
        if alpha > 0:
            x, y = make_correlated_xy(corr, size=902629,
                                      seed=rs.integers(MSEED))
            # smooth correlated noise vectors + project to surface
            xs = create_surface_grf(noise=x, alpha=alpha, normalize=normalize)
            ys = create_surface_grf(noise=y, alpha=alpha, normalize=normalize)
        else:
            xs, ys = make_correlated_xy(corr, size=20484,
                                        seed=rs.integers(MSEED))

        # remove medial wall to ensure data are still sufficiently correlated.
        # this is important for parcellations that will ignore the medial wall
        xs, ys = _mod_medial(xs, remove=True), _mod_medial(ys, remove=True)
        acorr = np.corrcoef(xs, ys)[0, 1]

        if debug:
            # n:>3 because dear lord i hope it doesn't take more than 999 tries
            print(f'{n:>3}: {acorr:>6.3f}')
            n += 1

    if acorr < 0:
        ys *= -1

    if normalize:
        xs, ys = sstats.zscore(xs), sstats.zscore(ys)

    return _mod_medial(xs, remove=False), _mod_medial(ys, remove=False)
