# -*- coding: utf-8 -*-
"""
Script for combining Moran's I outputs from simulated data
"""

from dataclasses import asdict, make_dataclass
from pathlib import Path

import pandas as pd

from netneurotools import stats as nnstats
from parspin import simnulls, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()
SimCorr = make_dataclass('simcorr', (
    'parcellation', 'scale', 'alpha', 'corr'
))
N_SIM = 1000


def calc_simcorr(parcellation, scale, alpha):
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

    # load simulated data
    alphadir = SIMDIR / alpha
    if parcellation == 'vertex':
        x, y = simnulls.load_vertex_data(alphadir, n_sim=N_SIM)
    else:
        x, y = simnulls.load_parc_data(alphadir, parcellation, scale,
                                       n_sim=N_SIM)
    corrs = nnstats.efficient_pearsonr(x, y, nan_policy='omit')[0]
    return pd.DataFrame(asdict(SimCorr(parcellation, scale, alpha, corrs)))


def main():
    df = []
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)
    for alpha in simnulls.ALPHAS:
        df.append(calc_simcorr('vertex', 'fsaverage5', alpha))
        for parcellation, annotations in parcellations.items():
            for scale in annotations:
                df.append(calc_simcorr(parcellation, scale, alpha))
    df = pd.concat(df, ignore_index=True)
    # this is gonna be a very big file... :man_shrugging:
    df.to_csv(SIMDIR / 'corr_summary.csv', index=False)


if __name__ == "__main__":
    main()
