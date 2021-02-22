# -*- coding: utf-8 -*-
"""
Script for running null (spatial permutation) models for correlation matrices
derived from NeuroSynth data. Saves out null distribution for each parcel /
resolution / spin method and prints out the number of correlations that survive
FWE correction.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import special

from brainspace.null_models import moran
from netneurotools import datasets as nndata, freesurfer as nnsurf
from parspin import utils as putils

ROIDIR = Path('./data/raw/rois').resolve()
NSDIR = Path('./data/derivatives/neurosynth').resolve()
SPDIR = Path('./data/derivatives/spins').resolve()
DISTDIR = Path('./data/derivatives/geodesic').resolve()
SURRDIR = Path('./data/derivatives/surrogates').resolve()
SPINTYPES = [
    'naive-nonpara',
    'vazquez-rodriguez',
    'baum',
    'cornblath',
    'vasa',
    'hungarian',
    'burt2018',
    'burt2020',
    'moran'
]
# alpha (FWE) level for assessing significance
ALPHA = 0.05


def _get_permcorr(data, perm):
    """
    Gets max value of correlation between `data` and `perm`

    Excludes diagonal of correlation

    Parameters
    ----------
    data : (R, T) array_like
        Input data where `R` is regions and `T` is neurosynth terms
    perm : (R, T) array_like
        Permuted data where `R` is regions and `T` is neurosynth terms

    Returns
    -------
    corr : float
        Maximum value of correlations between `data` and `perm`
    """

    data, perm = np.asarray(data), np.asarray(perm)

    # don't include NaN data in the correlation process
    mask = np.logical_not(np.all(np.isnan(perm), axis=1))

    # we want to correlat terms across regions, not vice-versa
    data, perm = data[mask].T, perm[mask].T
    out = np.corrcoef(data, perm)

    # grab the upper right quadrant of the resultant correlation matrix and
    # mask the diagonal, then take absolute max correlation
    mask_diag = np.logical_not(np.eye(len(data)), dtype=bool)
    corrs = out[len(data):, :len(data)] * mask_diag

    return np.abs(corrs).max()


def gen_permcorrs(data, spins, fname):
    """
    Generates permuted correlations for `data` with `spins`

    Parameters
    ----------
    data : (R, T) array_like
        Input data where `R` is regions and `T` is neurosynth terms
    spins : (R, P) array_like
        Spin resampling matrix where `R` is regions and `P` is the number of
        resamples
    fname : str or os.PathLike
        Filepath specifying where generated null distribution should be saved

    Returns
    -------
    perms : (P, 1) numpy.ndarray
        Permuted correlations
    """

    data = np.asarray(data)

    fname = putils.pathify(fname)
    if fname.exists():
        return np.loadtxt(fname).reshape(-1, 1)

    if isinstance(spins, (str, os.PathLike)):
        spins = np.loadtxt(spins, delimiter=',', dtype='int32')

    permcorrs = np.zeros((spins.shape[-1], 1))
    for n, spin in enumerate(spins.T):
        msg = f'{n:>5}/{spins.shape[-1]}'
        print(msg, end='\b' * len(msg), flush=True)
        # this will only have False values when spintype == 'baum'
        mask = np.logical_and(spin != -1, np.all(~np.isnan(data), axis=1))
        # get the absolute max correlation from the null correlation matrix
        permcorrs[n] = _get_permcorr(data[mask], data[spin][mask])

    print(' ' * len(msg) + '\b' * len(msg), end='', flush=True)

    # save these to disk for later re-use
    putils.save_dir(fname, permcorrs)

    return permcorrs


def get_fwe(real, perm, alpha=ALPHA):
    """
    Gets p-values from `real` based on null distribution of `perm`

    Parameters
    ----------
    real : (T, T) array_like
        Original correlation matrix (or similar)
    perm : (P, 1) array_like
        Null distribution for `real` based on `P` permutations

    Returns
    -------
    pvals : (T, T) array_like
        Non-parametric p-values for `real`
    """

    real, perm = np.asarray(real), np.asarray(perm)

    if perm.ndim == 1:
        perm = perm.reshape(-1, 1)

    pvals = np.sum(perm >= np.abs(real.flatten()), axis=0) / len(perm)
    pvals = np.reshape(pvals, real.shape)

    thresh = np.sum(np.triu(pvals < alpha, k=1))
    print(f'{thresh:>4} correlation(s) survive FWE-correction')

    return pvals


def get_surrogates(data, surrdir, scale):
    """
    Returns surrogate-reordered `data`

    Parameters
    ----------
    data : (R, T) pandas.DataFrame
        Input data where `R` is regions and `T` is neurosynth terms. Must have
        valid column names for loading surrogate data
    surrdir : str or os.PathLike
        Directory where surrogate resampling arrays are kept
    scale : str
        Scale of parcellation to be used (determines which surrogates to load)

    Returns
    -------
    surrogates : (R, T, P) numpy.ndarray
        Re-ordered `data` for `P` surrogates
    """

    # separately sort left / right hemispheres (surrogates were generated for
    # each hemisphere independently)
    idx_lh = [n for n, f in enumerate(data.index) if 'lh_' in f]
    idx_rh = [n for n, f in enumerate(data.index) if 'rh_' in f]
    stacked_data = pd.DataFrame(np.vstack((
        np.asarray(data)[idx_lh],
        np.asarray(data)[idx_rh]
    )), columns=data.columns)

    # sniff the number of surrogates that were generated to pre-generate the
    # output data array
    n_surr = np.loadtxt(surrdir / data.columns[0] / f'{scale}_surrogates.csv',
                        delimiter=',', skiprows=len(data) - 1).size
    permdata = np.empty((*data.shape, n_surr))
    for n, concept in enumerate(stacked_data.columns):
        # load surrogate indices for given concept
        surr_idx = np.loadtxt(surrdir / concept / f'{scale}_surrogates.csv',
                              delimiter=',', dtype='int32')
        permdata[:, n, :] = np.asarray(stacked_data[concept])[surr_idx]

    return permdata


def load_data(parcellation, scale):
    """
    Loads data for specified `parcellation` and `scale`

    Parameters
    ----------
    parcellation : {'atl-cammoun2012', 'atl-schaefer2018', 'vertex'}
        Name of parcellation to use
    scale : str
        Scale of parcellation to use. Must be valid scale for specified `parc`

    Returns
    -------
    nsdata : pd.DataFrame
        Loaded dataframe, where each column is a unique NeuroSynth term
    """

    # load data for provided `parcellation` and `scale`
    nsdata = pd.read_csv(NSDIR / parcellation / f'{scale}.csv', index_col=0)

    # drop the corpus callosum / unknown / medial wall parcels, if present
    todrop = np.array(putils.DROP)[np.isin(putils.DROP, nsdata.index)]
    if len(todrop) > 0:
        nsdata = nsdata.drop(todrop, axis=0)

    # if we're using vertex-based spins, set the medial wall to NaN
    if parcellation == 'vertex':
        labdir = ROIDIR / 'tpl-fsaverage' / 'fsaverage5' / 'label'
        medial = np.hstack([
            nib.freesurfer.read_label(labdir / 'lh.Medial_wall.label')
            for hemi, add in zip(['lh', 'rh'], [0, 10242])
        ])
        nsdata.loc[medial] = np.nan

    return nsdata


def run_null(parcellation, scale, spintype):
    """
    Runs spatial permutation null model for given combination of inputs

    Parameters
    ----------
    parcellation : str
        Name of parcellation to be used
    scale : str
        Scale of `parcellation` to be used
    spintype : str
        Name of spin method to be used

    Returns
    -------
    stats : pd.DataFrame
        Generated statistics with columns ['parcellation', 'scale', 'spintype',
        'n_sig']
    """

    nsdata = load_data(parcellation, scale)

    # run the damn thing
    print(f'Running {spintype:>9} spins for {scale}: ', end='', flush=True)
    out = NSDIR / parcellation / 'nulls' / spintype / f'{scale}_nulls.csv'
    if out.exists():
        permcorrs = np.loadtxt(out).reshape(-1, 1)
    elif spintype == 'cornblath':
        # even though we're working with parcellated data we need to project
        # that to the surface + spin the vertices, so let's load our
        # pre-generated vertex-level spins
        spins = SPDIR / 'vertex' / 'vazquez-rodriguez' / 'fsaverage5_spins.csv'

        # get annotation files
        fetcher = getattr(nndata, f"fetch_{parcellation.replace('atl-', '')}")
        annotations = fetcher('fsaverage5', data_dir=ROIDIR)[scale]

        # pre-load the spins for this function (assumes `spins` is array)
        # permdata will be an (R, T, n_rotate) array
        print('Pre-loading spins...', end='\b' * 20, flush=True)
        spins = np.loadtxt(spins, delimiter=',', dtype='int32')
        permdata = nnsurf.spin_data(nsdata, version='fsaverage5',
                                    lhannot=annotations.lh,
                                    rhannot=annotations.rh,
                                    spins=spins, n_rotate=spins.shape[-1],
                                    verbose=True)
        permcorrs = np.vstack([
            _get_permcorr(nsdata, permdata[..., n])
            for n in range(permdata.shape[-1])
        ])
        putils.save_dir(out, permcorrs)
    elif spintype in ['burt2018', 'burt2020']:
        surrdir = SURRDIR / parcellation / spintype / 'neurosynth'
        # generate the permuted data from the surrogate resampling arrays
        print('Generating surrogates...', end='\b' * 24, flush=True)
        permdata = get_surrogates(nsdata, surrdir, scale)
        permcorrs = np.vstack([
            _get_permcorr(nsdata, permdata[..., n])
            for n in range(permdata.shape[-1])
        ])
        putils.save_dir(out, permcorrs)
    elif spintype == 'moran':
        surrogates = np.zeros((*nsdata.shape, 10000))
        for hemi, dist, idx in putils.yield_data_dist(DISTDIR, parcellation,
                                                      scale, nsdata):
            mrs = moran.MoranRandomization(joint=True, n_rep=10000,
                                           tol=1e-6, random_state=1234)
            mrs.fit(dist)
            surrogates[idx] = mrs.randomize(hemi).transpose(1, 2, 0)

        permcorrs = np.vstack([
            _get_permcorr(nsdata, surrogates[..., n])
            for n in range(surrogates.shape[-1])
        ])
        putils.save_dir(out, permcorrs)
    else:
        spins = SPDIR / parcellation / spintype / f'{scale}_spins.csv'
        permcorrs = gen_permcorrs(nsdata, spins, out)

    nsdata = nsdata.dropna(axis=0, how='all')
    pvals = get_fwe(np.corrcoef(nsdata.T), permcorrs)

    out = pd.DataFrame(dict(
        parcellation=parcellation,
        scale=scale,
        spintype=spintype,
        n_sig=np.sum(np.triu(pvals < ALPHA, k=1))
    ), index=[0])

    return out


def main():
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    # output dataframe
    cols = ['parcellation', 'scale', 'spintype', 'n_sig']
    data = pd.DataFrame(columns=cols)

    # let's run all our parcellations
    for parcellation, annotations in parcellations.items():
        print(f'PARCELLATION: {parcellation}')
        for scale in annotations:
            for spintype in SPINTYPES:
                data = data.append(run_null(parcellation, scale, spintype),
                                   ignore_index=True)

            # now calculate parametric null
            nsdata = load_data(parcellation, scale, spintype)
            corrs = np.corrcoef(nsdata.T)
            # this calculates the correlation value for p < ALPHA cutoff
            ab = (len(nsdata) / 2) - 1
            cutoff = 1 - (special.btdtri(ab, ab, ALPHA / 2) * 2)
            # now add the parametric null to our giant summary dataframe
            data = data.append(pd.DataFrame({
                'parcellation': parcellation,
                'scale': scale,
                'spintype': 'naive-para',
                'n_sig': np.sum(np.triu(corrs > cutoff, k=1))
            }, index=[0]), ignore_index=True)

    data.to_csv(NSDIR / 'summary.csv', index=False)


if __name__ == "__main__":
    main()
