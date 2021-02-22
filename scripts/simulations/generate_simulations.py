# -*- coding: utf-8 -*-
"""
Generates simulated (matching) GRFs on the surface of fsaverage5 with a
specified correlation for use in "ground truth" testing of spatial null models.
"""

from pathlib import Path

from joblib import Parallel, delayed
import nibabel as nib
import numpy as np
import pandas as pd

from parspin.spatial import matching_multinorm_grfs
from parspin import simnulls, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
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

    # checkpointing in case of restarts
    if (outdir / f'x_{seed:04d}.mgh').exists():
        return

    x, y = matching_multinorm_grfs(corr=corr, alpha=alpha, seed=seed)
    for data, name in zip((x, y), ('x', 'y')):
        img = nib.freesurfer.mghformat.MGHImage(
            data.astype('float32'), affine=None
        )
        fn = outdir / f'{name}_{seed:04d}.mgh'
        nib.save(img, fn)


def parcellate_sim(val, alphadir, annot):
    """
    Parcellates simulated surface GRF with `annot`

    Parameters
    ----------
    val : {'x', 'y'}
        Which simulated vector to parcellate
    alphadir : os.PathLike
        Directory in which simulated data are stored
    annot : (2,) namedtuple
        With entries ('lh', 'rh') of filepaths to annotation files to be used
        to parcellate data

    Returns
    -------
    data : (N, `N_SIM`) pandas.DataFrame
        Where `N` is the number of regions in the parcellation and the index
        of the dataframe are the region names
    """

    data = pd.DataFrame(index=putils.get_names(lh=annot.lh, rh=annot.rh))
    alpha = alphadir.parent.name
    for sim in putils.trange(N_SIM, desc=f'Parcellating {alpha} {val}'):
        img = nib.load(alphadir / f'{val}_{sim:04d}.mgh').get_fdata().squeeze()
        cdata = []
        for n, hemi in enumerate(('lh', 'rh')):
            sl = slice(10242 * n, 10242 * (n + 1))
            cdata.append(putils.parcellate(img[sl], getattr(annot, hemi)))
        data = data.assign(**{str(sim): np.hstack(cdata)})

    return data


if __name__ == '__main__':
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    for alpha in simnulls.ALPHAS:
        outdir = SIMDIR / f'alpha-{float(alpha):.1f}' / 'sim'
        outdir.mkdir(parents=True, exist_ok=True)

        # generate simulated GRFs
        Parallel(n_jobs=N_PROC)(
            delayed(create_and_save_grfs)(
                corr=0.15, alpha=alpha, seed=n, outdir=outdir
            )
            for n in putils.trange(N_SIM, desc=f'Simulating alpha-{alpha:.2f}')
        )

        # parcellate simulated GRFs and save as CSV
        for name, annotations in parcellations.items():
            for scale, annot in annotations.items():
                scdir = outdir.parent / name
                scdir.mkdir(parents=True, exist_ok=True)
                for val in ('x', 'y'):
                    fn = scdir / f'{scale}_{val}.csv'
                    if fn.exists():
                        continue
                    parcellate_sim(val, outdir, annot).to_csv(fn, sep=',')
