# -*- coding: utf-8 -*-
"""
Script for combining Moran's I outputs from simulated data
"""

from dataclasses import asdict, make_dataclass
from pathlib import Path
import time

import pandas as pd
import numpy as np

from parspin import simnulls, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()

N_SIM = 10000  # number of simulations to calculate empirical Moran's I
N_PROC = 24  # number of parallel workers for surrogate generation
SEED = 1234  # reproducibility
Moran = make_dataclass('Moran', (
    'parcellation', 'scale', 'alpha', 'spatnull', 'sim', 'moran'
))


def calc_moran(parcellation, scale, alpha):
    """
    Calculate's Moran's I of all simulations for provided inputs

    Parameters
    ----------
    parcellation : str
        Name of parcellation to be used
    scale : str
        Scale of `parcellation` to be used
    alpha : float
        Spatial autocorrelation parameter to be used

    Returns
    -------
    moran_fn : os.PathLike
        Path to generated file containing Moran's I for simulations
    """

    print(f'{time.ctime()}: {parcellation} {scale} {alpha}', flush=True)

    # filename for output
    moran_fn = (SIMDIR / alpha / parcellation / f'{scale}_moran.csv')

    if moran_fn.exists():
        return moran_fn

    # load simulated data
    alphadir = SIMDIR / alpha
    if parcellation == 'vertex':
        y = simnulls.load_vertex_data(alphadir, n_sim=N_SIM)[1]
    else:
        y = simnulls.load_parc_data(alphadir, parcellation, scale,
                                    n_sim=N_SIM)[1]

    dist = simnulls.load_full_distmat(y, DISTDIR, parcellation, scale)
    moran = simnulls.calc_moran(dist, np.asarray(y), n_jobs=N_PROC)
    putils.save_dir(moran_fn, np.atleast_1d(moran), overwrite=False)

    return moran_fn


def combine_moran(parcellation, scale, alpha):
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

    Returns
    -------
    df : pd.DataFrame
        With columns ['parcellation', 'scale', 'alpha', 'spantull', 'sim',
        'moran']
    """

    # filename for output
    mfn = calc_moran(parcellation, scale, alpha)
    morani = np.loadtxt(mfn)
    df = pd.DataFrame(
        asdict(Moran(
            parcellation, scale, alpha, 'empirical', range(len(morani)), morani
        ))
    )
    for spatnull in simnulls.SPATNULLS:
        if spatnull == 'naive-para':
            continue
        if spatnull not in simnulls.VERTEXWISE and parcellation == 'vertex':
            continue
        mfn = (SIMDIR / alpha / parcellation / 'nulls' / spatnull / 'pvals')
        morani = np.loadtxt(mfn / f'{scale}_moran_9999.csv')
        df = df.append(
            pd.DataFrame(asdict(
                Moran(parcellation, scale, alpha, spatnull, '9999', morani)
            )), ignore_index=True
        )

    return df


def main():
    df = []
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)
    for alpha in simnulls.ALPHAS:
        df.append(combine_moran('vertex', 'fsaverage5', alpha))
        for parcellation, annotations in parcellations.items():
            for scale in annotations:
                df.append(combine_moran(parcellation, scale, alpha))
    col = ['parcellation', 'scale', 'alpha', 'spatnull', 'sim', 'moran']
    df = pd.concat(df, ignore_index=True)[col]
    df.to_csv(SIMDIR / 'moran_summary.csv', index=False)


if __name__ == "__main__":
    main()
