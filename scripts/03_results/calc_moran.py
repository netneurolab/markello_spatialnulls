# -*- coding: utf-8 -*-
"""
Script for calculating Moran's I for original simulated data
"""

from pathlib import Path
import time

import numpy as np

from parspin import simnulls, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SPDIR = Path('./data/derivatives/spins').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()

N_SIM = 10000  # number of simulations to load
N_PROC = 24  # number of parallel workers for surrogate generation
SEED = 1234  # reproducibility


def calc_moran(parcellation, scale, alpha):
    """
    Runs spatial null models for given combination of inputs

    Parameters
    ----------
    parcellation : str
        Name of parcellation to be used
    scale : str
        Scale of `parcellation` to be used
    alpha : float
        Spatial autocorrelation parameter to be used
    """

    print(f'{time.ctime()}: {parcellation} {scale} {alpha}', flush=True)

    # filename for output
    moran_fn = (SIMDIR / alpha / parcellation / f'{scale}_moran.csv')

    # load simulated data
    alphadir = SIMDIR / alpha
    if parcellation == 'vertex':
        y = simnulls.load_vertex_data(alphadir, n_sim=N_SIM)[1]
    else:
        y = simnulls.load_parc_data(alphadir, parcellation, scale,
                                    n_sim=N_SIM)[1]

    if not moran_fn.exists():
        dist = simnulls.load_full_distmat(y, DISTDIR, parcellation, scale)
        moran = simnulls.calc_moran(dist, np.asarray(y), n_jobs=N_PROC)
        putils.save_dir(moran_fn, np.atleast_1d(moran), overwrite=True)


def main():
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)
    for alpha in ['alpha-0.0']:
        calc_moran('vertex', 'fsaverage5', alpha)
        for parcellation, annotations in parcellations.items():
            for scale in annotations:
                calc_moran(parcellation, scale, alpha)


if __name__ == "__main__":
    main()
