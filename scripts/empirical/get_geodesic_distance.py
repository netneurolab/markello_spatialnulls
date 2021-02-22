# -*- coding: utf-8 -*-
"""
Calculates parcel-parcel geodesic distance matrices for all parcellations. The
vertex-vertex distance matrix is also calculated for the `fsaverage5` surface.
Generated matrices are saved in `data/derivatives/geodesic`.
"""

from pathlib import Path

from netneurotools import datasets as nndata, freesurfer as nnsurf
from parspin import surface, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
N_PROC = 24  # parallelization of distance calculation
SURFACE = 'pial'  # surface on which to calculate distance

if __name__ == '__main__':
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)
    surf = nndata.fetch_fsaverage('fsaverage5', data_dir=ROIDIR)[SURFACE]
    subj, spath = nnsurf.check_fs_subjid('fsaverage5')
    medial = Path(spath) / subj / 'label'
    medial_labels = [
        'unknown', 'corpuscallosum', '???',
        'Background+FreeSurfer_Defined_Medial_Wall'
    ]

    # get parcel distance matrices with this horrible nested for-loop :scream:
    for name, annotations in parcellations.items():
        for scale, annot in annotations.items():
            for hemi in ('lh', 'rh'):
                for allow_med in (True, False):
                    med = 'medial' if allow_med else 'nomedial'
                    out = DISTDIR / name / med / f'{scale}_{hemi}_dist.csv'
                    if out.exists():
                        continue
                    # when we want to disallow travel along the medial wall we
                    # can specify which labels in our parcellation belong to
                    # the medial wall and disallow travel along vertices
                    # belonging to those parcels
                    mlabels = None if allow_med else medial_labels
                    dist = surface.get_surface_distance(getattr(surf, hemi),
                                                        getattr(annot, hemi),
                                                        medial_labels=mlabels,
                                                        n_proc=N_PROC,
                                                        use_wb=False,
                                                        verbose=True)
                    putils.save_dir(out, dist)

    # get vertex distance matrix
    for hemi in ('lh', 'rh'):
        medial_path = medial / f'{hemi}.Medial_wall.label'
        for allow_med in (True, False):
            med = 'medial' if allow_med else 'nomedial'
            out = DISTDIR / 'vertex' / med / f'fsaverage5_{hemi}_dist.csv'
            if out.exists():
                continue
            mpath = None if allow_med else medial_path
            # since we have no parcellation here we need to provide a file
            # that denotes which vertices belong to the medial wall (`mpath`)
            dist = surface.get_surface_distance(getattr(surf, hemi),
                                                medial=mpath,
                                                n_proc=N_PROC,
                                                use_wb=False,
                                                verbose=True)
            putils.save_dir(out, dist)
