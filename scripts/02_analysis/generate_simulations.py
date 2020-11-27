# -*- coding: utf-8 -*-
"""
Generates simulated (matching) GRFs on the surface of fsaverage5 with a
specified correlation for use in "ground truth" testing of spatial null models.
"""

from pathlib import Path

from joblib import Parallel, delayed
import nibabel as nib
import numpy as np

from parspin.spatial import matching_multinorm_grfs
from parspin import utils as putils

SIMDIR = Path('./data/derivatives/simulated').resolve()
N_PROC = 36
N_SIM = 10000


def create_and_save_grfs(corr, alpha, seed, outdir):
    """
    Generates matching GRFs on fsaverage5 surface + saves outputs as MGH files

    Parameters
    ----------
    corr : float
        Desired correlation of GRFs
    alpha : float
        Desired spatial autocorrelation of GRFS
    seed : int, None, randomState
        Random state seed
    outdir : pathlib.Path
        Where generated GRFs should be saved
    """

    x, y = matching_multinorm_grfs(corr=corr, alpha=alpha, seed=seed)
    for data, name in zip((x, y), ('x', 'y')):
        img = nib.freesurfer.mghformat.MGHImage(
            data.astype('float32'), affine=None
        )
        fn = outdir / f'{name}_{str(seed).zfill(4)}.mgh'
        nib.save(img, fn)


if __name__ == '__main__':
    for alpha in np.arange(0, 3.5, 0.5):
        outdir = SIMDIR / 'alpha-{:.1f}'.format(float(alpha))
        outdir.mkdir(parents=True, exist_ok=True)

        # generate
        Parallel(n_jobs=N_PROC)(
            delayed(create_and_save_grfs)(
                corr=0.15, alpha=alpha, seed=n, outdir=outdir
            )
            for n in putils.trange(N_SIM, desc=f'Alpha {alpha}')
        )
