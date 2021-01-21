# -*- coding: utf-8 -*-
"""
Generates figures examining regional probability of duplicate resamplings for
Vazquez-Rodriguez and Baum methods
"""

from pathlib import Path

import numpy as np
from parspin import plotting, utils as putils

FIGDIR = Path('./figures/spins/probabilities').resolve()
ROIDIR = Path('./data/raw/rois').resolve()
SPINDIR = Path('./data/derivatives/spins').resolve()


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)

    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    for name, annotations in parcellations.items():
        print(f'PARCELLATION: {name}')
        for sptype in ['vazquez-rodriguez', 'baum']:
            for scale, annot in annotations.items():
                print(f'Assessing probabilities for {sptype} ({scale})')
                # load spins and calculate proportion of times each region is
                # reassigned across all spins (normalized by number of spins)
                # if prob > 1, region is assigned more than it "should" be
                # if prob < 1, region is assigned less than it "should" be
                fname = SPINDIR / name / sptype / f'{scale}_spins.csv'
                spins = np.loadtxt(fname, delimiter=',', dtype='int32')
                probs = np.unique(spins, return_counts=True)[-1]
                probs = probs / spins.shape[-1]
                # generate and save brain plot
                out = FIGDIR / name / sptype / f'{scale}.png'
                plotting.save_brainmap(probs, out, annot.lh, annot.rh,
                                       subject_id='fsaverage5',
                                       views=['lat', 'med'])
