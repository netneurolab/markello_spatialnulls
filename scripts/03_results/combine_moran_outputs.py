# -*- coding: utf-8 -*-
"""
Script for combining Moran's I outputs from simulated data
"""

from dataclasses import asdict, make_dataclass
from pathlib import Path

import pandas as pd
import numpy as np

from parspin import simnulls, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()
Moran = make_dataclass('Moran', (
    'parcellation', 'scale', 'alpha', 'spatnull', 'moran'
))


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
    stats : dict
        With keys 'parcellation', 'scale', 'spatnull', 'alpha', and 'prob',
        where 'prob' is the probability that the p-value for a given simulation
        is less than ALPHA (across all simulations)
    """

    # filename for output
    mfn = (SIMDIR / alpha / parcellation / f'{scale}_moran.csv')
    morani = np.loadtxt(mfn)
    df = pd.DataFrame(asdict(
        Moran(parcellation, scale, alpha, 'empirical', morani)
    ))
    for spatnull in simnulls.SPATNULLS:
        if spatnull == 'naive-para':
            continue
        if spatnull not in simnulls.VERTEXWISE and parcellation == 'vertex':
            continue
        mfn = (SIMDIR / alpha / parcellation / 'nulls' / spatnull / 'pvals')
        morani = np.loadtxt(mfn / f'{scale}_moran_9999.csv')
        df = df.append(
            pd.DataFrame(asdict(
                Moran(parcellation, scale, alpha, spatnull, morani)
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
    df = pd.concat(df, ignore_index=True)
    # this is gonna be a very big file... :man_shrugging:
    df.to_csv(SIMDIR / 'moran_summary.csv', index=False)


if __name__ == "__main__":
    main()
