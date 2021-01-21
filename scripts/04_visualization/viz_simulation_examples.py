# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

from netneurotools.freesurfer import FSIGNORE
from parspin import spatial, simnulls, utils
from parspin.plotting import save_brainmap

FIGSIZE = 500
ROIDIR = Path('./data/raw/rois').resolve()
FIGDIR = Path('./figures/simulated/examples').resolve()

OPTS = dict(
    subject_id='fsaverage5',
    views=['lat', 'med']
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
        if not fn.exists():
            save_brainmap(sim, fn, **OPTS)

        for parc, annots in parcellations.items():
            for scale, annot in annots.items():
                # parcellate simulation according
                data = np.hstack([
                    utils.parcellate(np.split(sim, 2)[n], hemi, FSIGNORE)
                    for n, hemi in enumerate(annot)
                ])
                # save plot of parcellated simulation
                fn = FIGDIR / alpha / parc / f'{scale}.png'
                if not fn.exists():
                    save_brainmap(data, fn, lh=annot.lh, rh=annot.rh, **OPTS)
