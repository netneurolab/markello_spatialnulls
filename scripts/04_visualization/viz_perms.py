# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from brainsmash.mapgen import Base
from brainspace.null_models import moran
from netneurotools import (datasets as nndata,
                           freesurfer as nnsurf,
                           utils as nnutils)
from parspin import burt
from parspin.plotting import save_brainmap
# from parspin.utils import PARULA

FIGSIZE = 500
SPINDIR = Path('./data/derivatives/spins').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
ROIDIR = Path('./data/raw/rois').resolve()
FIGDIR = Path('./figures/spins/examples').resolve()
OPTS = {
    'colorbar': False,
    'colormap': 'coolwarm',
    'vmin': 0
}
warnings.simplefilter('ignore', category=np.VisibleDeprecationWarning)


if __name__ == "__main__":
    cammoun = nndata.fetch_cammoun2012('MNI152NLin2009aSym', data_dir=ROIDIR)
    name, scale = 'atl-cammoun2012', 'scale125'
    lh, rh = nndata.fetch_cammoun2012('fsaverage', data_dir=ROIDIR)[scale]
    info = pd.read_csv(cammoun['info'])
    info = info.query(f'scale == "{scale}" & structure == "cortex"')
    n_right = len(info.query('hemisphere == "R"'))

    # get coordinates and make LH/RH like surface info
    labels = np.asarray(info['id'])
    coords = nnutils.get_centroids(cammoun[scale], labels=labels,
                                   image_space=True)
    coords = np.row_stack([coords[n_right:], coords[:n_right]])

    # generate re-ordering of coordinates based on Y-position
    start = end = 0
    data = np.zeros((len(coords)))
    order = np.zeros(len(coords), dtype=int)
    inds = np.arange(len(coords), dtype=int)
    for i in np.asarray(info.groupby('hemisphere').count()['id']):
        end += i
        c = coords[start:end]
        order[start:end] = inds[c[:, 1].argsort()[::-1] + start]
        data[start:end] = np.arange(i)
        start = i
    np.put(data, order, data)
    OPTS['vmax'] = len(data) - n_right

    # plot original data and save
    save_brainmap(data, FIGDIR / 'raw_surf.png', lh, rh, **OPTS)

    spins = ['naive-nonpara', 'vazquez-rodriguez', 'vasa', 'hungarian', 'baum']
    for sptype in spins:
        spin = np.loadtxt(SPINDIR / name / sptype / f'{scale}_spins.csv',
                          delimiter=',', dtype=int, usecols=0)
        plot = np.append(data, [np.nan])[spin]  # accounts for 'baum' (-1 idx)
        save_brainmap(plot, FIGDIR / f'{sptype}_surf.png', lh, rh, **OPTS)

    # cornblath
    spin = np.loadtxt(SPINDIR / 'vertex' / 'vazquez-rodriguez'
                      / 'fsaverage5_spins.csv', delimiter=',', dtype=int,
                      usecols=0)[:, None]
    lh5, rh5 = nndata.fetch_cammoun2012('fsaverage5')[scale]
    plot = nnsurf.spin_data(data, lhannot=lh5, rhannot=rh5, spins=spin,
                            n_rotate=1)
    save_brainmap(plot, FIGDIR / 'cornblath_surf.png', lh, rh, **OPTS)

    # burt 2018
    lhdata, rhdata = data[:end - start], data[end - start:]
    lhdist = np.loadtxt(DISTDIR / name / 'nomedial' / f'{scale}_lh_dist.csv',
                        delimiter=',')
    rhdist = np.loadtxt(DISTDIR / name / 'nomedial' / f'{scale}_rh_dist.csv',
                        delimiter=',')
    plot = np.hstack((
        burt.make_surrogate(lhdist, lhdata + 1, seed=1234),
        burt.make_surrogate(rhdist, rhdata + 1, seed=1234)
    ))
    save_brainmap(plot, FIGDIR / 'burt2018_surf.png', lh, rh, **OPTS)

    # burt 2020 (need to rescale to original data range)
    plot = np.hstack((
        nnutils.rescale(Base(lhdata, lhdist, seed=1234)(200, 50).T[:, 180],
                        lhdata.min(), lhdata.max()),
        nnutils.rescale(Base(rhdata, rhdist, seed=1234)(200, 50).T[:, 180],
                        rhdata.min(), rhdata.max())
    ))
    save_brainmap(plot, FIGDIR / 'burt2020_surf.png', lh, rh, **OPTS)

    # moran spectral randomization
    np.fill_diagonal(lhdist, 1)
    np.fill_diagonal(rhdist, 1)
    lhdist **= -1
    rhdist **= -1
    mrs = moran.MoranRandomization(joint=True, n_rep=1000, tol=1e-6,
                                   random_state=1234)
    plot = np.hstack((
        np.squeeze(mrs.fit(lhdist).randomize(lhdata)),
        np.squeeze(mrs.fit(rhdist).randomize(rhdata))
    ))[611]
    save_brainmap(plot, FIGDIR / 'moran_surf.png', lh, rh, **OPTS)
