# -*- coding: utf-8 -*-
"""
Script for running null (spatial permutation) models to test partition
specificity of the T1w/T2w ratio within both the Yeo instrinsic functional
networks and the von Economo & Koskinas cytoarchitectonic classes
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats

from brainspace.null_models import moran
from netneurotools import datasets as nndata, freesurfer as nnsurf
from parspin.partitions import NET_OPTIONS, NET_CODES
from parspin import simnulls, utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
HCPDIR = Path('./data/derivatives/hcp').resolve()
SPDIR = Path('./data/derivatives/spins').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
SURRDIR = Path('./data/derivatives/surrogates').resolve()
SPINTYPES = [
    'naive-para',
    'naive-nonpara',
    'vazquez-rodriguez',
    'vasa',
    'hungarian',
    'baum',
    'cornblath',
    'burt2018',
    'burt2020',
    'moran'
]
# alpha (FWE) level for assessing significance
ALPHA = 0.05
# percent of parcels in a given network that can be dropped (due to e.g.,
# medial wall rotation) before that spin is discarded. for example, if
# PCTDROPTHRESH = 1 then _all_ parcels must be missing in a given network
# before that spin will be discarded, whereas if PCTDROPTHRESH = 0.25 then only
# >25% of the parcels in a given network need to be missing for that spin to be
# discarded
# n.b., this should only impact the 'baum' and 'cornblath' methods!
PCTDROPTHRESH = 1


def _get_netmeans(data, networks, nets=None):
    """
    Gets average of `data` within each label specified in `networks`

    Parameters
    ----------
    data : (N,) array_like
        Data to be averaged within networks
    networks : (N,) array_like
        Network label for each entry in `data`

    Returns
    -------
    means : (L,) numpy.ndarray
        Means of networks
    """

    data, networks = np.asarray(data), np.asarray(networks)
    nparc = np.bincount(networks)[1:]

    if nets is None:
        nets = np.trim_zeros(np.unique(networks))

    mask = np.logical_not(np.isnan(data))

    # if there are too many nans in a given network, don't use this spin
    pct_dropped = 1 - (ndimage.sum(mask, networks, nets) / nparc)
    if np.any(pct_dropped >= PCTDROPTHRESH):
        return np.full(len(nets), np.nan)

    # otherwise, compute the average T1w/T2w within each network
    data, networks = data[mask], networks[mask]
    with np.errstate(invalid='ignore'):
        permnets = ndimage.mean(data, networks, nets)

    return permnets


def gen_permnets(data, networks, spins, fname):
    """
    Generates permuted network partitions of `data` and `networks` with `spins`

    Parameters
    ----------
    data : (R,) array_like
        Input data where `R` is regions
    networks : (R,) array_like
        Network labels for `R` regions
    spins : (R, P) array_like
        Spin resampling matrix where `R` is regions and `P` is the number of
        resamples
    fname : str or os.PathLike
        Filepath specifying where generated null distribution should be saved

    Returns
    -------
    permnets : (P, L) numpy.ndarray
        Permuted network means for `L` networks
    """

    data, networks = np.asarray(data), np.asarray(networks)

    # if the output file already exists just load that and return it
    fname = Path(fname)
    if fname.exists():
        return np.loadtxt(fname, delimiter=',')

    # if we were given a file for the resampling array, load it
    if isinstance(spins, (str, os.PathLike)):
        spins = simnulls.load_spins(spins, n_perm=10000)

    nets = np.trim_zeros(np.unique(networks))
    permnets = np.full((spins.shape[-1], len(nets)), np.nan)
    for n, spin in enumerate(spins.T):
        msg = f'{n:>5}/{spins.shape[-1]}'
        print(msg, end='\b' * len(msg), flush=True)

        spindata = data[spin]
        spindata[spin == -1] = np.nan

        # get the means of each network for each spin
        permnets[n] = _get_netmeans(spindata, networks, nets)

    print(' ' * len(msg) + '\b' * len(msg), end='', flush=True)
    putils.save_dir(fname, permnets)

    return permnets


def get_fwe(real, perm, alpha=ALPHA):
    """
    Gets p-values from `real` based on null distribution of `perm`

    Parameters
    ----------
    real : (1, L) array_like
        Real partition means for `L` networks
    perm : (S, L) array_like
        Null partition means for `S` permutations of `L` networks
    alpha : (0, 1) float, optional
        Alpha at which to check for p-value significance. Default: ALPHA

    Returns
    -------
    zscores : (L,) numpy.ndarray
        Z-scores of `L` networks
    pvals : (L,) numpy.ndarray
        P-values corresponding to `L` networks
    """

    real, perm = np.asarray(real), np.asarray(perm)

    if real.ndim == 1:
        real = real.reshape(1, -1)

    # de-mean distributions to get accurate two-tailed p-values
    permmean = np.nanmean(perm, axis=0, keepdims=True)
    permstd = np.nanstd(perm, axis=0, keepdims=True, ddof=1)
    real -= permmean
    perm -= permmean

    # get z-scores and pvals (add 1 to numerator / denominator for pvals)
    zscores = np.squeeze(real / permstd)
    numerator = np.sum(np.abs(np.nan_to_num(perm)) >= np.abs(real), axis=0)
    denominator = np.sum(np.logical_not(np.isnan(perm)), axis=0)
    pvals = np.squeeze((1 + numerator) / (1 + denominator))

    # print networks with pvals below threshold
    print(', '.join([f'{z:.2f}' if pvals[n] < ALPHA else '0.00'
                     for n, z in enumerate(zscores)]))

    return zscores, pvals


def get_surrogates(data, surrdir, scale):
    """
    Returns surrogate-reordered `data`

    Parameters
    ----------
    data : (R,) pandas.DataFrame
        Input data where `R` is regions
    surrdir : str or os.PathLike
        Directory where surrogate resampling arrays are kept
    scale : str
        Scale of parcellation to be used (determines which surrogates to load)

    Returns
    -------
    surrogates : (R, P) numpy.ndarray
        Re-ordered `data` for `P` surrogates
    """

    # separately sort left / right hemispheres (surrogates were generated for
    # each hemisphere independently)
    idx_lh = [n for n, f in enumerate(data.index) if 'lh_' in f]
    idx_rh = [n for n, f in enumerate(data.index) if 'rh_' in f]
    stacked_data = np.hstack((
        np.asarray(data)[idx_lh],
        np.asarray(data)[idx_rh]
    ))

    # load the surrogate data resampling arrays and return the shuffled data
    surr_idx = np.loadtxt(surrdir / f'{scale}_surrogates.csv', delimiter=',',
                          dtype='int')

    return stacked_data[surr_idx]


def load_data(netclass, parc, scale):
    """
    Parameters
    ----------
    netclass : {'yeo', 'vek'}
        Network classes to use
    parc : {'atl-cammoun2012', 'atl-schaefer2018', 'vertex'}
        Name of parcellation to use
    scale : str
        Scale of parcellation to use. Must be valid scale for specified `parc`

    Returns
    -------
    data : pandas.DataFrame
        Loaded dataframe with columns 'myelin' and 'networks'
    """

    # load data for provided `parcellation` and `scale`
    data = pd.read_csv(HCPDIR / parc / f'{scale}.csv', index_col=0)

    # get the RSN affiliations for the provided parcellation + scale
    networks = NET_OPTIONS[netclass][parc](scale)
    if parc == 'vertex':
        # we want the vertex-level affiliations if we have vertex data
        data = data.assign(networks=getattr(networks, 'vertices'))
        # when working with vertex-level data, our spins were generated with
        # the medial wall / corpuscallosum included, but we need to set these
        # to NaN so they're ignored in the final sums
        data.loc[data['networks'] == 0, 'myelin'] = np.nan
    else:
        # get the parcel-level affiliations if we have parcellated data
        data = data.assign(networks=getattr(networks, 'parcels'))
        # when working with parcellated data, our spins were NOT generated with
        # the medial wall / corpuscallosum included, so we should drop these
        # parcels (which should [ideally] have values of ~=0)
        todrop = np.array(putils.DROP)[np.isin(putils.DROP, data.index)]
        if len(todrop) > 0:
            data = data.drop(todrop, axis=0)

    return data


def run_null(netclass, parc, scale, spintype):
    """
    Runs spatial permutation null model for given combination of inputs

    Parameters
    ----------
    netclass : {'vek', 'yeo'}
        Network partition to test
    parc : str
        Name of parcellation to be used
    scale : str
        Scale of `parcellation` to be used
    spintype : str
        Name of spin method to be used

    Returns
    -------
    stats : pd.DataFrame
        Generated statistics with columns ['parcellation', 'scale', 'spintype',
        'netclass', 'network', 'zscore', 'pval']
    """
    data = load_data(netclass, parc, scale)

    # run the damn thing
    print(f'Running {spintype:>9} spins for {scale}: ', end='', flush=True)
    out = (HCPDIR / parc / 'nulls' / netclass / spintype
           / f'thresh{PCTDROPTHRESH * 100:03.0f}' / f'{scale}_nulls.csv')
    if out.exists():
        permnets = np.loadtxt(out, delimiter=',')
    elif spintype == 'naive-para':
        data['myelin'] = stats.zscore(data['myelin'], ddof=1)
        data = data.query('networks != 0')
        pvals = np.asarray([
            stats.ttest_1samp(data.loc[idx, 'myelin'], 0)[-1]
            for net, idx in data.groupby('networks').groups.items()
        ])
        zscores = np.asarray(data.groupby('networks').mean()['myelin'])
        # print networks with pvals below threshold
        print(', '.join([f'{z:.2f}' if pvals[n] < ALPHA else '0.00'
                        for n, z in enumerate(zscores)]))
    elif spintype == 'cornblath':
        # even though we're working with parcellated data we need to project
        # that to the surface + spin the vertices, so let's load our
        # pre-generated vertex-level spins
        spins = SPDIR / 'vertex' / 'vazquez-rodriguez' / 'fsaverage5_spins.csv'

        # get annotation files (we need these to project parcels to surface)
        fetcher = getattr(nndata, f"fetch_{parc.replace('atl-', '')}")
        annotations = fetcher('fsaverage5', data_dir=ROIDIR)[scale]

        # pre-load the spins for this function (assumes `spins` is array)
        spins = simnulls.load_spins(spins, n_perm=10000)
        # generate "spun" data; permdata will be an (R, T, n_rotate) array
        # where `R` is regions and `T` is 1 (myelination)
        permdata = nnsurf.spin_data(np.asarray(data['myelin']),
                                    version='fsaverage5',
                                    lhannot=annotations.lh,
                                    rhannot=annotations.rh,
                                    spins=spins, n_rotate=spins.shape[-1],
                                    verbose=True)
        permnets = np.vstack([
            _get_netmeans(permdata[..., n], data['networks'])
            for n in range(spins.shape[-1])
        ])
        putils.save_dir(out, permnets)
    elif spintype in ['burt2018', 'burt2020']:
        surrdir = SURRDIR / parc / spintype / 'hcp'
        surrogates = get_surrogates(data['myelin'], surrdir, scale)
        permnets = np.vstack([
            _get_netmeans(surrogates[..., n], data['networks'])
            for n in range(surrogates.shape[-1])
        ])
        putils.save_dir(out, permnets)
    elif spintype == 'moran':
        surrogates = np.zeros((len(data['myelin']), 10000))
        for hemi, dist, idx in putils.yield_data_dist(DISTDIR, parc,
                                                      scale, data['myelin']):
            mrs = moran.MoranRandomization(joint=True, n_rep=10000,
                                           tol=1e-6, random_state=1234)
            mrs.fit(dist)
            surrogates[idx] = np.squeeze(mrs.randomize(hemi)).T

        permnets = np.vstack([
            _get_netmeans(surrogates[..., n], data['networks'])
            for n in range(surrogates.shape[-1])
        ])
        putils.save_dir(out, permnets)
    else:
        spins = SPDIR / parc / spintype / f'{scale}_spins.csv'
        permnets = gen_permnets(data['myelin'], data['networks'],
                                spins, out)

    # now get the real network averages and compare to the permuted values
    if spintype != 'naive-para':
        real = _get_netmeans(data['myelin'], data['networks'])
        zscores, pvals = get_fwe(real, permnets)

    out = pd.DataFrame(dict(
        parcellation=parc,
        scale=scale,
        spintype=spintype,
        netclass=netclass,
        network=list(NET_CODES[netclass].keys()),
        zscore=zscores,
        pval=pvals
    ))

    return out


def main():
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    # output dataframe
    data = pd.DataFrame(columns=[
        'parcellation', 'scale', 'spintype', 'netclass',
        'network', 'zscore', 'pval'
    ])

    # use both Yeo + VEK network groupings
    for netclass in ['yeo', 'vek']:
        # run all our parcellations + spintypes
        for parcellation, annotations in parcellations.items():
            print(f'PARCELLATION: {parcellation}')
            for scale in annotations.keys():
                for spintype in SPINTYPES:
                    data = data.append(run_null(netclass, parcellation,
                                                scale, spintype),
                                       ignore_index=True)

    # save the output data !
    data.to_csv(HCPDIR / f'summary_thresh{PCTDROPTHRESH * 100:03.0f}.csv.gz',
                index=False)


if __name__ == "__main__":
    main()
