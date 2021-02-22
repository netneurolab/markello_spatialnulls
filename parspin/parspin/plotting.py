# -*- coding: utf-8 -*-
"""
Functions for plotting
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from nilearn.plotting import plot_surf_stat_map
import numpy as np

from netneurotools.datasets import fetch_fsaverage
from netneurotools.plotting import plot_fsaverage, plot_fsvertex
from parspin.utils import pathify


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
    data = np.nan_to_num(data)
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


def save_brainmap(data, fname, lh=None, rh=None, **kwargs):
    """
    Plots `data` to the surface and saves to `fname`

    Parameters
    ----------
    plot : array_like
        Parcellated data to be plotted to the surface. Should be in the order
        {left,right} hemisphere
    fname : str or os.PathLike
        Filepath where plotted figure should be saved
    {lh,rh} : str or os.pathLike, optional
        Annotation files for the {left,right} hemisphere if `data` are
        in parcellated format. By default assumes files are 'fsaverage'
        resolution; set `subject_id` kwarg if different. Default: None
    kwargs : key-value pairs
        Passed to :func:`netneurotools.plotting.plot_fsvertex` (default) or
        :func:`netneurotools.plotting.plot_fsaverage` (if data are parcellated)

    Returns
    -------
    fname : os.PathLike
        Same as provided `fname`
    """

    if (lh is not None and rh is None) or (lh is None and rh is not None):
        raise ValueError('Both lh and rh must be provided')

    opts = dict(
        alpha=1.0, views=['lat'], colormap='RdBu_r', colorbar=True,
        surf='inflated', subject_id='fsaverage', size_per_view=500,
        offscreen=True,
    )
    opts.update(kwargs)

    if lh is None and rh is None:
        fig = plot_fsvertex(data, **opts)
    else:
        fig = plot_fsaverage(data, lhannot=lh, rhannot=rh, **opts)

    pathify(fname).parent.mkdir(parents=True, exist_ok=True)
    fig.save_image(fname)
    fig.close()

    return fname


def savefig(fig, fname):
    """
    Saves `fig` to `fname`, creating parent directories if necessary

    Parameters
    ----------
    fig : matplotlib.pyplot.Figure
        Figure object to be saved
    fname : str or os.PathLike
        Filepath to where `fig` should be saved
    """

    fname = pathify(fname)
    if not fname.parent.exists():
        fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname, bbox_inches='tight', transparent=True)
    plt.close(fig=fig)
