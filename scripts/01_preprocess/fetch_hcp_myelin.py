# -*- coding: utf-8 -*-
"""
Script for mapping HCP group-average myelination data to the `fsaverage5`
surface. Assumes you've downloaded the S1200 group-averaged myelin map and
placed it---without renaming it---in the `data/raw/hcp` folder. We cannot
provide this file as users must sign the HCP Data Agreement in order to gain
access.

Once mapped to the `fsaverage5` surface data are parcellated using both
Schaefer et al., 2018 and Cammoun et al., 2012 atlases
"""

from pathlib import Path

import pandas as pd
import nibabel as nib
import numpy as np

from parspin import utils as putils
from netneurotools import datasets as nndata
from netneurotools.utils import run

HCPDIR = Path('./data/raw/hcp').resolve()
ROIDIR = Path('./data/raw/rois').resolve()
PARDIR = Path('./data/derivatives/hcp').resolve()

# command for separating CIFTI file into hemisphere-specific GIFTI files
CIFTISEP = 'wb_command -cifti-separate {cifti} COLUMN ' \
           '-metric CORTEX_LEFT {lhout} -metric CORTEX_RIGHT {rhout}'
# command for resampling 32k_fs_LR group data to fsaverage5 group space
HCP2FS = 'wb_command -metric-resample {gii} ' \
         '{path}/resample_fsaverage/' \
         'fs_LR-deformed_to-fsaverage.{hemi}.sphere.32k_fs_LR.surf.gii ' \
         '{path}/resample_fsaverage/' \
         'fsaverage5_std_sphere.{hemi}.10k_fsavg_{hemi}.surf.gii ' \
         'ADAP_BARY_AREA {out} -area-metrics ' \
         '{path}/resample_fsaverage/' \
         'fs_LR.{hemi}.midthickness_va_avg.32k_fs_LR.shape.gii ' \
         '{path}/resample_fsaverage/' \
         'fsaverage5.{hemi}.midthickness_va_avg.10k_fsavg_{hemi}.' \
         'shape.gii'
# command for converting GIFTI to MGH
GIITOMGH = 'mris_convert -f {gii} {surf} {out}'


def get_hcp_data(fname, annot=None):
    """
    Generates dataframe for (parcellated) HCP T1w/T2w data

    If `annot` is provided dataframe will be parcellated data; otherwise,
    vertex-level data will be saved

    Parameters
    ----------
    fname : str or os.PathLike
        Filepath where generated data should be saved
    annot : (2,) namedtuple
        With entries ('lh', 'rh') of filepaths to annotation files to be used
        to parcellate data. If not provided will yield vertex-level data.
        Default: None

    Returns
    -------
    fname : os.PathLike
        Path to generated data
    """

    index = None
    if annot is not None:
        index = putils.get_names(lh=annot.lh, rh=annot.rh)

    cdata = []
    for hemi in ['lh', 'rh']:
        mgh = HCPDIR / f'{hemi}.myelin.mgh'
        if annot is not None:
            mgh = putils.parcellate(mgh, getattr(annot, hemi))
        else:
            mgh = np.squeeze(nib.load(mgh).dataobj)
        cdata.append(mgh)
    data = pd.DataFrame(dict(myelin=np.hstack(cdata)), index=index)

    fname.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(fname, sep=',')

    return fname


if __name__ == "__main__":
    stds = nndata.fetch_hcp_standards(data_dir=ROIDIR)
    fsaverage = nndata.fetch_fsaverage('fsaverage5', data_dir=ROIDIR)['sphere']

    # separate cifti into hemispheres (and convert to gifti)
    cifti = HCPDIR / 'S1200.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
    lhout = HCPDIR / 'S1200.L.MyelinMap_BC_MSMSAll.32k_fs_LR.func.gii'
    rhout = HCPDIR / 'S1200.R.MyelinMap_BC_MSMSAll.32k_fs_LR.func.gii'
    run(CIFTISEP.format(cifti=cifti, lhout=lhout, rhout=rhout), quiet=True)

    # for each hemisphere, resample to FreeSurfer fsaverage5 space and convert
    # the resulting GII file to MGH (for consistency with NeuroSynth data)
    for gii, hemi, surf in zip((lhout, rhout), ('L', 'R'), fsaverage):
        out = HCPDIR / f'fsaverage5.MyelinMap.{hemi}.10k_fsavg_{hemi}.func.gii'
        mgh = HCPDIR / f'{hemi.lower()}h.myelin.mgh'
        run(HCP2FS.format(gii=gii, path=stds, hemi=hemi, out=out), quiet=True)
        run(GIITOMGH.format(gii=out, surf=surf, out=mgh), quiet=True)

        # remove intermediate file
        if out.exists():
            out.unlink()

    # remove intermediate files
    for fn in [lhout, rhout]:
        if fn.exists():
            fn.unlink()

    # get parcellations that we'll use to parcellate data
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    # get vertex-level data first
    get_hcp_data(PARDIR / 'vertex' / 'fsaverage5.csv')

    # now do parcellations
    for name, annotations in parcellations.items():
        for scale, annot in annotations.items():
            get_hcp_data(PARDIR / name / f'{scale}.csv', annot=annot)
