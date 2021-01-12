# -*- coding: utf-8 -*-
"""
Script to combine outputs of `run_simulated_nulls_serial.py` once all nulls +
simulations have completed.
"""

from pathlib import Path

import numpy as np

from parspin import simnulls, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()

N_PERM = 1000  # number of permutations for null models
N_SIM = 1000  # number of simulations to run


def combine_nulls(parcellation, scale, spatnull, alpha):
    """
    Combines outputs of all simulations into single files for provided inputs

    Parameters
    ----------
    parcellation : str
        Name of parcellation to be used
    scale : str
        Scale of `parcellation` to be used
    spatnull : str
        Name of spin method to be used
    alpha : float
        Spatial autocorrelation parameter to be used
    """

    nulldir = SIMDIR / alpha / parcellation / 'nulls' / spatnull
    pvals_fn = nulldir / f'{scale}_nulls.csv'
    perms_fn = nulldir / f'{scale}_perms.csv'

    pvals, perms = np.zeros(N_SIM), np.zeros((N_PERM, N_SIM))
    for sim in range(N_SIM):
        pvals[sim] = \
            np.loadtxt(nulldir / 'pvals' / f'{scale}_nulls_{sim:04d}.csv')
        perms[:, sim] = \
            np.loadtxt(nulldir / 'pvals' / f'{scale}_perms_{sim:04d}.csv')

    putils.save_dir(pvals_fn, pvals, overwrite=False)
    putils.save_dir(perms_fn, perms, overwrite=False)


def main():
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)
    for spatnull in simnulls.SPATNULLS:
        for alpha in simnulls.ALPHAS:
            if spatnull in simnulls.VERTEXWISE:
                combine_nulls('vertex', 'fsaverage5', spatnull, alpha)
            for parcellation, annotations in parcellations.items():
                for scale in annotations:
                    combine_nulls(parcellation, scale, spatnull, alpha)


if __name__ == "__main__":
    main()
