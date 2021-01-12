# -*- coding: utf-8 -*-
"""
Script for generating spatial permutation resampling arrays for a variety of
different methods.
"""

from pathlib import Path

import numpy as np

from netneurotools import freesurfer as nnsurf, stats as nnstats
from parspin import utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
SPINDIR = Path('./data/derivatives/spins').resolve()


if __name__ == '__main__':
    # get cammoun + schaefer parcellations
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    # generate the vertex-level spins
    coords, hemi = nnsurf._get_fsaverage_coords('fsaverage5', 'sphere')

    fname = SPINDIR / 'vertex' / 'vazquez-rodriguez' / 'fsaverage5_spins.csv'
    if not fname.exists():
        print('Generating V-R spins for fsaverage5 surface')
        spins = nnsurf.gen_spinsamples(coords, hemi, exact=False,
                                       n_rotate=10000, verbose=True,
                                       seed=1234, check_duplicates=False)
        putils.save_dir(fname, spins)

    fname = SPINDIR / 'vertex' / 'naive-nonpara' / 'fsaverage5_spins.csv'
    if not fname.exists():
        print('Generating naive permutations for fsaverage5 surface')
        rs = np.random.default_rng(1234)
        spins = np.column_stack([
            rs.permutation(len(coords)) for f in range(10000)
        ])
        putils.save_dir(fname, spins)

    # now pre-generate the parcellation spins for five methods. we can't
    # pre-generate the project-reduce-average method because that relies on the
    # data itself, but we _can_ use the above vertex-level spins for that
    for name, annotations in parcellations.items():
        print(f'PARCELLATION: {name}')
        for scale, annot in annotations.items():
            coords, hemi = nnsurf.find_parcel_centroids(lhannot=annot.lh,
                                                        rhannot=annot.rh,
                                                        version='fsaverage5',
                                                        surf='sphere',
                                                        method='surface')
            spin_fn = f'{scale}_spins.csv'

            fname = SPINDIR / name / 'vazquez-rodriguez' / spin_fn
            if not fname.exists():
                print(f'Generating V-R spins for {scale}')
                spins = nnstats.gen_spinsamples(coords, hemi, exact=False,
                                                n_rotate=10000, verbose=True,
                                                check_duplicates=False,
                                                seed=1234)
                putils.save_dir(fname, spins)

            fname = SPINDIR / name / 'vasa' / spin_fn
            if not fname.exists():
                print(f'Generating Vasa spins for {scale}')
                spins = nnstats.gen_spinsamples(coords, hemi, exact='vasa',
                                                n_rotate=10000, verbose=True,
                                                check_duplicates=False,
                                                seed=1234)
                putils.save_dir(fname, spins)

            fname = SPINDIR / name / 'hungarian' / spin_fn
            if not fname.exists():
                print(f'Generating Hungarian spins for {scale}')
                spins = nnstats.gen_spinsamples(coords, hemi, exact=True,
                                                n_rotate=10000, verbose=True,
                                                check_duplicates=False,
                                                seed=1234)
                putils.save_dir(fname, spins)

            fname = SPINDIR / name / 'baum' / spin_fn
            if not fname.exists():
                print(f'Generating Baum spins for {scale}')
                spins = nnsurf.spin_parcels(lhannot=annot.lh, rhannot=annot.rh,
                                            version='fsaverage5', seed=1234,
                                            n_rotate=10000, verbose=True,
                                            check_duplicates=False)
                putils.save_dir(fname, spins)

            fname = SPINDIR / name / 'naive-nonpara' / spin_fn
            if not fname.exists():
                print(f'Generating naive permutations for {scale}')
                rs = np.random.default_rng(1234)
                spins = np.column_stack([
                    rs.permutation(len(coords)) for f in range(10000)
                ])
                putils.save_dir(fname, spins)
