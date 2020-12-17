# -*- coding: utf-8 -*-
"""
Functions for plotting
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from nilearn.plotting import plot_surf_stat_map
import numpy as np

from netneurotools.datasets import fetch_fsaverage
from netneurotools.plotting import plot_fsaverage
from parspin.utils import PARULA


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

    data = np.split(np.nan_to_num(data), 2)
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
    fig = plot_fsaverage(data, lhannot=lh, rhannot=rh, **opts)
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.save_image(fname)
    fig.close()
