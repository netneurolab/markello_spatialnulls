# -*- coding: utf-8 -*-
"""
Tests computational time of different null methods and plots outputs
"""

from dataclasses import asdict, make_dataclass
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import threadpoolctl

from brainsmash import mapgen
from brainspace.null_models import moran
from netneurotools import (datasets as nndata,
                           freesurfer as nnsurf,
                           stats as nnstats)
from parspin import burt, simnulls, surface
from parspin.plotting import savefig
from parspin.simnulls import SPATNULLS
from parspin.utils import PARCHUES, SPATHUES

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.sans-serif'] = ['Myriad Pro']

DATADIR = Path('./data/derivatives/supplementary/comp_time').resolve()
FIGDIR = Path('./figures/supplementary/comp_time').resolve()
ORDER = ('vertex', 'atl-cammoun2012', 'atl-schaefer2018')
ROIDIR = Path('./data/raw/rois').resolve()
SPDIR = Path('./data/derivatives/spins').resolve()
SIMDIR = Path('./data/derivatives/simulated').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
OUTDIR = Path('./data/derivatives/supplementary/comp_time').resolve()

ALPHA = 'alpha-2.0'
N_PERM = 1000
N_REPEAT = 5
SEED = 1234
USE_CACHED = True
PARCS = (
    ('vertex', 'fsaverage5'),
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
        fn = SPDIR / 'vertex' / 'vazquez-rodriguez' / 'fsaverage5_spins.csv'
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
        fn = DISTDIR / parcellation / 'nomedial' / f'{scale}_{hemi}_dist.npy'
        dist = np.load(fn, allow_pickle=False, mmap_mode='c').astype('float32')
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
                                                use_wb=False,
                                                verbose=True)
        else:
            annot = _get_annot(parcellation, scale)
            dist = surface.get_surface_distance(getattr(surf, hemi),
                                                getattr(annot, hemi),
                                                medial_labels=medial_labels,
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
            idx = np.asarray([
                n for n, f in enumerate(data.index)if f.startswith(hemi)
            ])
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
            # Box-Cox transformation requires positive data
            hdata += np.abs(dmin) + 0.1
            surrogates[idx] = \
                burt.batch_surrogates(dist, hdata, n_surr=N_PERM, seed=SEED)
        elif spatnull == 'burt2020':
            if parcellation == 'vertex':
                index = np.argsort(dist, axis=-1)
                dist = np.sort(dist, axis=-1)
                surrogates[idx] = \
                    mapgen.Sampled(hdata, dist, index, seed=SEED)(N_PERM).T
            else:
                surrogates[idx] = \
                    mapgen.Base(hdata, dist, seed=SEED)(N_PERM, 50).T
        elif spatnull == 'moran':
            dist = dist.astype('float64')  # required for some reason...
            np.fill_diagonal(dist, 1)
            dist **= -1
            mrs = moran.MoranRandomization(joint=True, n_rep=N_PERM,
                                           tol=1e-6, random_state=SEED)
            surrogates[idx] = mrs.fit(dist).randomize(hdata).T

    return surrogates


def output_exists(data, parcellation, scale, spatnull, repeat):
    """
    Checks whether given combination of inputs already exists in `data`

    Returns
    -------
    exits : bool
        Whether outputs have already been run (True) or not (False)
    """

    if len(data) == 0:
        return False

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    present = data.query(f'parcellation == "{parcellation}" '
                         f'& scale == "{scale}" '
                         f'& spatnull == "{spatnull}"')
    return len(present) > (repeat)


def make_stripplot(fn):
    """
    Makes stripplot of runtime for different spatial null models

    Parameters
    ----------
    fn : {'cached.csv', 'uncached.csv'}
        Filename to load runtime data from

    Returns
    -------
    ax : matplotlib.pyplot.Axes
        Axis with plot
    """
    data = pd.read_csv(DATADIR / fn)
    fig, ax = plt.subplots(1, 1)
    ax = sns.stripplot(x='runtime', y='spatnull', hue='parcellation',
                       order=SPATNULLS, hue_order=ORDER, dodge=True, ax=ax,
                       data=data, palette=PARCHUES)
    ax.set_xscale('log')
    xl = (10**-3, 10**5)
    ax.hlines(np.arange(0.5, 9.5), *xl, linestyle='dashed', linewidth=0.5,
              color=np.array([50, 50, 50]) / 255)
    ax.set(xlim=xl, ylim=(9.5, -0.5))
    yticklabels = ax.get_yticklabels()
    for n, ytick in enumerate(yticklabels):
        ytick.set_color(SPATHUES[n])
    ax.set_yticklabels(yticklabels)
    ax.legend_.set_visible(False)
    sns.despine(ax=ax)

    return ax


def compute_all_runtimes():
    # limit multi-threading; NO parallelization
    threadpoolctl.threadpool_limits(limits=1)
    # output
    fn = OUTDIR / ('cached.csv' if USE_CACHED else 'uncached.csv')
    fn.parent.mkdir(exist_ok=True, parents=True)

    cols = ['parcellation', 'scale', 'spatnull', 'runtime']
    data = pd.read_csv(fn).to_dict('records') if fn.exists() else []
    for spatnull in simnulls.SPATNULLS:
        for parc, scale in PARCS:
            if parc == "vertex" and spatnull not in simnulls.VERTEXWISE:
                continue
            for repeat in range(N_REPEAT):
                if output_exists(data, parc, scale, spatnull, repeat):
                    continue
                data.append(get_runtime(parc, scale, spatnull))
                pd.DataFrame(data)[cols].to_csv(fn, index=False)

    return fn


if __name__ == "__main__":
    for cache in (True, False):
        globals()['USE_CACHED'] = cache
        fn = compute_all_runtimes()
        ax = make_stripplot(fn)
        savefig(ax.figure, FIGDIR / f'{fn.name[:-4]}.svg')
