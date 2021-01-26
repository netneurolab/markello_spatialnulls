# -*- coding: utf-8 -*-

from pathlib import Path

import nibabel as nib
import numpy as np

from netneurotools.freesurfer import FSIGNORE
from parspin import spatial, simnulls, utils
from parspin.plotting import save_brainmap

FIGSIZE = 500
ROIDIR = Path('./data/raw/rois').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()
FIGDIR = Path('./figures/simulated/examples').resolve()
SAVE_ALL = False
OVERWRITE = True
OPTS = dict(
    subject_id='fsaverage5',
    views=['lat', 'med'],
    colormap='coolwarm'
)

if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    parcellations = utils.get_cammoun_schaefer(data_dir=ROIDIR)

    for alpha in simnulls.ALPHAS:
        # create simulation with desired alpha (same seed as all others)
        sim = spatial.create_surface_grf(alpha=float(alpha[-3:]), seed=1234,
                                         medial_val=np.nan)
        # save plot of vertex-wise simulation
        fn = FIGDIR / alpha / 'vertex' / 'fsaverage5.png'
        if not fn.exists() or OVERWRITE:
            save_brainmap(sim, fn, **OPTS)

        # save plot of parcellated simulations (if desired)
        if SAVE_ALL:
            for parc, annots in parcellations.items():
                for scale, annot in annots.items():
                    # parcellate simulation according
                    data = np.hstack([
                        utils.parcellate(np.split(sim, 2)[n], hemi, FSIGNORE)
                        for n, hemi in enumerate(annot)
                    ])
                    # save plot of parcellated simulation
                    fn = FIGDIR / alpha / parc / f'{scale}.png'
                    if not fn.exists() or OVERWRITE:
                        save_brainmap(data, fn, lh=annot.lh, rh=annot.rh,
                                      **OPTS)

        # now load one pair of correlated simulations and save
        for img in ('x', 'y'):
            fn = FIGDIR / alpha / 'sim' / f'{img}_9999.png'
            if not fn.exists() or OVERWRITE:
                img = nib.load(SIMDIR / alpha / 'sim' / f'{img}_9999.mgh')
                data = np.squeeze(img.get_fdata())
                data[data == 0] = np.nan
                save_brainmap(data, fn, **OPTS)
