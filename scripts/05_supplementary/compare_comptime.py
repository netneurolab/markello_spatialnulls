# -*- coding: utf-8 -*-
"""
Tests computational time of different null methods
"""

from dataclasses import asdict, make_dataclass
import time
from pathlib import Path

import numpy as np
import pandas as pd
import threadpoolctl

from brainsmash import mapgen
from brainspace.null_models import moran
from netneurotools import (datasets as nndata,
                           freesurfer as nnsurf,
                           stats as nnstats)
from parspin import burt, simnulls, surface

ROIDIR = Path('./data/raw/rois').resolve()
SPDIR = Path('./data/derivatives/spins').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
OUTDIR = Path('./data/derivatives/supplementary/comp_time').resolve()

ALPHA = 'alpha-2.0'
N_PROC = 1
N_PERM = 1000
N_REPEAT = 5
SEED = 1234
USE_CACHED = False
PARCS = (
    ('atl-cammoun2012', 'scale500'),
    ('atl-schaefer2018', '1000Parcels7Networks')
)
CompTime = make_dataclass(
    'CompTime', ('parcellation', 'scale', 'spatnull', 'runtime')
)


def get_runtime(parcellation, scale, spatnull):
    """
    Runs spatial null models for given combination of inputs

    Parameters
    ----------
    parcellation : str
        Name of parcellation to be used
    scale : str
        Scale of `parcellation` to be used
    spatnull : str
        Name of spin method to be used
    """

    # filenames (for I/O)
    fn = SPDIR / parcellation / spatnull / f'{scale}_spins.csv'

    # load simulated data
    alphadir = SIMDIR / ALPHA
    if parcellation == 'vertex':
        x, y = simnulls.load_vertex_data(alphadir, sim=0)
    else:
        x, y = simnulls.load_parc_data(alphadir, parcellation, scale, sim=0)

    # start timer (after loading data--accounds for diff b/w vertex/parc)
    start = time.time()

    # calculate the null p-values
    if spatnull == 'naive-para':
        nnstats.efficient_pearsonr(x, y, nan_policy='omit')[1]
        nulls = None
    elif spatnull == 'naive-nonpara':
        nulls = naive_nonpara(y, fn=fn)
    elif spatnull == 'vazquez-rodriguez':
        nulls = vazquez_rodriguez(y, parcellation, scale, fn=fn)
    elif spatnull == 'vasa':
        nulls = vasa(y, parcellation, scale, fn=fn)
    elif spatnull == 'hungarian':
        nulls = hungarian(y, parcellation, scale, fn=fn)
    elif spatnull == 'cornblath':
        nulls = cornblath(y, parcellation, scale, fn=fn)
    elif spatnull == 'baum':
        nulls = baum(y, parcellation, scale, fn=fn)
    elif spatnull in ('burt2018', 'burt2020', 'moran'):
        nulls = make_surrogates(y, parcellation, scale, spatnull, fn=fn)
    else:
        raise ValueError(f'Invalid spatnull: {spatnull}')

    if nulls is not None:
        simnulls.calc_pval(x, y, nulls)

    end = time.time()
    ct = CompTime(parcellation, scale, spatnull, end - start)
    print(ct)

    return asdict(ct)


def _get_annot(parcellation, scale):
    fetcher = getattr(nndata, f"fetch_{parcellation.replace('atl-', '')}")
    return fetcher('fsaverage5', data_dir=ROIDIR)[scale]


def naive_nonpara(y, fn=None):
    y = np.asarray(y)
    rs = np.random.default_rng(SEED)
    if USE_CACHED and fn is not None:
        spins = simnulls.load_spins(fn, n_perm=N_PERM)
    else:
        spins = np.column_stack([
            rs.permutation(len(y)) for f in range(N_PERM)
        ])
    return y[spins]


def vazquez_rodriguez(y, parcellation, scale, fn=None):
    y = np.asarray(y)
    if USE_CACHED and fn is not None:
        spins = simnulls.load_spins(fn, n_perm=N_PERM)
    else:
        if parcellation != 'vertex':
            annot = _get_annot(parcellation, scale)
            coords, hemi = nnsurf.find_parcel_centroids(lhannot=annot.lh,
                                                        rhannot=annot.rh,
                                                        version='fsaverage5',
                                                        surf='sphere',
                                                        method='surface')
        else:
            coords, hemi = nnsurf._get_fsaverage_coords(scale, 'sphere')
        spins = nnstats.gen_spinsamples(coords, hemi, method='original',
                                        n_rotate=N_PERM, seed=SEED)
    return y[spins]


def vasa(y, parcellation, scale, fn=None):
    y = np.asarray(y)
    if USE_CACHED and fn is not None:
        spins = simnulls.load_spins(fn, n_perm=N_PERM)
    else:
        annot = _get_annot(parcellation, scale)
        coords, hemi = nnsurf.find_parcel_centroids(lhannot=annot.lh,
                                                    rhannot=annot.rh,
                                                    version='fsaverage5',
                                                    surf='sphere',
                                                    method='surface')
        spins = nnstats.gen_spinsamples(coords, hemi, method='vasa',
                                        n_rotate=N_PERM, seed=SEED)
    return y[spins]


def hungarian(y, parcellation, scale, fn=None):
    y = np.asarray(y)
    if USE_CACHED and fn is not None:
        spins = simnulls.load_spins(fn, n_perm=N_PERM)
    else:
        annot = _get_annot(parcellation, scale)
        coords, hemi = nnsurf.find_parcel_centroids(lhannot=annot.lh,
                                                    rhannot=annot.rh,
                                                    version='fsaverage5',
                                                    surf='sphere',
                                                    method='surface')
        spins = nnstats.gen_spinsamples(coords, hemi, method='hungarian',
                                        n_rotate=N_PERM, seed=SEED)
    return y[spins]


def baum(y, parcellation, scale, fn=None):
    y = np.asarray(y)
    if USE_CACHED and fn is not None:
        spins = simnulls.load_spins(fn, n_perm=N_PERM)
    else:
        annot = _get_annot(parcellation, scale)
        spins = nnsurf.spin_parcels(lhannot=annot.lh, rhannot=annot.rh,
                                    version='fsaverage5', n_rotate=N_PERM,
                                    seed=SEED)
    nulls = y[spins]
    nulls[spins == -1] = np.nan
    return nulls


def cornblath(y, parcellation, scale, fn=None):
    y = np.asarray(y)
    annot = _get_annot(parcellation, scale)
    spins = simnulls.load_spins(fn, n_perm=N_PERM) if USE_CACHED else None
    nulls = nnsurf.spin_data(y, version='fsaverage5', spins=spins,
                             lhannot=annot.lh, rhannot=annot.rh,
                             n_rotate=N_PERM, seed=SEED)
    return nulls


def get_distmat(hemi, parcellation, scale, fn=None):
    if hemi not in ('lh', 'rh'):
        raise ValueError(f'Invalid hemishere designation {hemi}')

    if USE_CACHED and fn is not None:
        fn = DISTDIR / parcellation / 'medial' / f'{scale}_{hemi}_dist.csv'
        npy = fn.with_suffix('.npy')
        if npy.exists():
            dist = np.load(npy, allow_pickle=False, mmap_mode='c')
        else:
            dist = np.loadtxt(fn, delimiter=',')
    else:
        surf = nndata.fetch_fsaverage('fsaverage5', data_dir=ROIDIR)['pial']
        subj, spath = nnsurf.check_fs_subjid('fsaverage5')
        medial = Path(spath) / subj / 'label'
        medial_labels = [
            'unknown', 'corpuscallosum', '???',
            'Background+FreeSurfer_Defined_Medial_Wall'
        ]
        if parcellation == 'vertex':
            medial_path = medial / f'{hemi}.Medial_wall.label'
            dist = surface.get_surface_distance(getattr(surf, hemi),
                                                medial=medial_path,
                                                n_proc=N_PROC,
                                                use_wb=False,
                                                verbose=True)
        else:
            annot = _get_annot(parcellation, scale)
            dist = surface.get_surface_distance(getattr(surf, hemi),
                                                getattr(annot, hemi),
                                                medial_labels=medial_labels,
                                                n_proc=N_PROC,
                                                use_wb=False,
                                                verbose=True)
    return dist


def make_surrogates(data, parcellation, scale, spatnull, fn=None):
    if spatnull not in ('burt2018', 'burt2020', 'moran'):
        raise ValueError(f'Cannot make surrogates for null method {spatnull}')

    darr = np.asarray(data)
    dmin = darr[np.logical_not(np.isnan(darr))].min()

    surrogates = np.zeros((len(data), N_PERM))
    for n, hemi in enumerate(('lh', 'rh')):
        dist = get_distmat(hemi, parcellation, scale, fn=fn)
        try:
            idx = [n for n, f in enumerate(data.index)if f.startswith(hemi)]
            hdata = np.squeeze(np.asarray(data.iloc[idx]))
        except AttributeError:
            idx = np.arange(n * (len(data) // 2), (n + 1) * (len(data) // 2))
            hdata = np.squeeze(data[idx])

        # handle NaNs before generating surrogates; should only be relevant
        # when using vertex-level data, but good nonetheless
        mask = np.logical_not(np.isnan(hdata))
        surrogates[idx[np.logical_not(mask)]] = np.nan
        hdata, dist, idx = hdata[mask], dist[np.ix_(mask, mask)], idx[mask]

        if spatnull == 'burt2018':
            # Box-Cox transformation requires positive data :man_facepalming:
            hdata += np.abs(dmin) + 0.1
            surrogates[idx] = \
                burt.batch_surrogates(dist, hdata, n_surr=N_PERM, seed=SEED)
        elif spatnull == 'burt2020':
            if parcellation == 'vertex':  # memmap is required for this shit
                index = np.argsort(dist, axis=-1)
                dist = np.sort(dist, axis=-1)
                surrogates[idx] = \
                    mapgen.Sampled(hdata, dist, index, seed=SEED)(N_PERM).T
            else:
                surrogates[idx] = \
                    mapgen.Base(hdata, dist, seed=SEED)(N_PERM, 50).T
        elif spatnull == 'moran':
            mrs = moran.MoranRandomization(joint=True, n_rep=N_PERM,
                                           tol=1e-6, random_state=SEED)
            surrogates[idx] = mrs.fit(dist).randomize(hdata).T

    return surrogates


def main():
    # limit multi-threading; NO parallelization
    threadpoolctl.threadpool_limits(limits=1)
    # output
    fn = OUTDIR / ('cached.csv' if USE_CACHED else 'uncached.csv')
    fn.parent.mkdir(exist_ok=True, parents=True)

    data = []
    for spatnull in simnulls.SPATNULLS:
        if spatnull in simnulls.VERTEXWISE:
            for repeat in range(N_REPEAT):
                data.append(get_runtime('vertex', 'fsaverage5', spatnull))
                pd.DataFrame(data).to_csv(fn, index=False)
        for parc, scale in PARCS:
            for repeat in range(N_REPEAT):
                data.append(get_runtime(parc, scale, spatnull))
                pd.DataFrame(data).to_csv(fn, index=False)


if __name__ == "__main__":
    main()
