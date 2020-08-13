# -*- coding: utf-8 -*-
"""
Script for performing NeuroSynth-style meta-analyses for all available
Cognitive Atlas concepts. Generated "association test" (i.e., reverse
inference) images for each meta-analysis are projected to the `fsaverage5`
surface with a sub-call to FreeSurfer's mri_vol2surf.

Once mapped to the `fsaverage5` surface data are parcellated using both
Schaefer et al., 2018 and Cammoun et al., 2012 atlases
"""

import contextlib
import json
import os
from pathlib import Path
import warnings

import pandas as pd
import nibabel as nib
import numpy as np
import requests

from parspin import utils as putils
from netneurotools.utils import run
import neurosynth as ns

# /sigh
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

ROIDIR = Path('./data/raw/rois').resolve()
NSDIR = Path('./data/raw/neurosynth').resolve()
PARDIR = Path('./data/derivatives/neurosynth').resolve()

# these are the images from the neurosynth analyses we'll save. we really only
# need the association-test_z but who cares about conserving disk space?
IMAGES = ['pA', 'pAgF', 'pFgA', 'association-test_z', 'uniformity-test_z']
# command for projecting MNI152 volumetric data to the fsaverage5 surface
VOL2SURF = 'mri_vol2surf --src {} --out {} --hemi {} --mni152reg ' \
           '--trgsubject fsaverage5 --projfrac 0.5 --interp nearest'


def fetch_ns_data(directory):
    """ Fetches NeuroSynth database + features to `directory`

    Paramerters
    -----------
    directory : str or os.PathLike
        Path to directory where data should be saved

    Returns
    -------
    database, features : PathLike
        Paths to downloaded NS data
    """

    directory = Path(directory)

    # if not already downloaded, download the NS data and unpack it
    database, features = directory / 'database.txt', directory / 'features.txt'
    if not database.exists() or not features.exists():
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            ns.dataset.download(path=directory, unpack=True)
        try:  # remove tarball if it wasn't removed for some reason
            (directory / 'current_data.tar.gz').unlink()
        except FileNotFoundError:
            pass

    return database, features


def get_cogatlas_concepts(url=None):
    """ Fetches list of concepts from the Cognitive Atlas

    Parameters
    ----------
    url : str
        URL to Cognitive Atlas API

    Returns
    -------
    concepts : set
        Unordered set of terms
    """

    if url is None:
        url = 'https://cognitiveatlas.org/api/v-alpha/concept'

    req = requests.get(url)
    req.raise_for_status()
    concepts = set([f.get('name') for f in json.loads(req.content)])

    return concepts


def run_meta_analyses(database, features, use_features=None, outdir=None):
    """
    Runs NS-style meta-analysis based on `database` and `features`

    Parameters
    ----------
    database, features : str or os.PathLike
        Path to NS-style database.txt and features.txt files
    use_features : list, optional
        List of features on which to run NS meta-analyses; if not supplied all
        terms in `features` will be used
    outdir : str or os.PathLike
        Path to output directory where derived files should be saved

    Returns
    -------
    generated : list of str
        List of filepaths to generated term meta-analysis directories
    """

    # check outdir
    if outdir is None:
        outdir = NSDIR
    outdir = Path(outdir)

    # make database and load feature names; annoyingly slow
    dataset = ns.Dataset(str(database))
    dataset.add_features(str(features))
    features = set(dataset.get_feature_names())

    # if we only want a subset of the features take the set intersection
    if use_features is not None:
        features = set(features) & set(use_features)
    pad = max([len(f) for f in features])

    generated = []
    for word in sorted(features):
        msg = f'Running meta-analysis for term: {word:<{pad}}'
        print(msg, end='\r', flush=True)

        # run meta-analysis + save specified outputs (only if they don't exist)
        path = outdir / word.replace(' ', '_')
        path.mkdir(exist_ok=True)
        if not all((path / f'{f}.nii.gz').exists() for f in IMAGES):
            ma = ns.MetaAnalysis(dataset, dataset.get_studies(features=word))
            ma.save_results(path, image_list=IMAGES)

        # project data to fsaverage5 surface and save mgh for each hemisphere
        for hemi in ['lh', 'rh']:
            fname = path / 'association-test_z.nii.gz'
            outname = path / f'{hemi}.association-test_z.mgh'
            run(VOL2SURF.format(fname, outname, hemi), quiet=True)

    print(' ' * len(msg) + '\b' * len(msg), end='', flush=True)

    return generated


def get_ns_data(generated, fname, annot=None):
    """
    Generates dataframe from `generated` NS meta-analyses and saves to `fname`

    If `annot` provided dataframe will be parcellated data

    Parameters
    ----------
    generated : (N,) list of os.PathLike
        Filepaths to outputs of NeuroSynth meta-analyses
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

    # empty dataframe to hold our data
    data = pd.DataFrame(index=index)
    for concept in generated:
        # get lh/rh for given concept and parcellate as applicable
        cdata = []
        for hemi in ('lh', 'rh'):
            mgh = concept / f'{hemi}.association-test_z.mgh'
            if annot is not None:
                mgh = putils.parcellate(mgh, getattr(annot, hemi))
            else:
                mgh = np.squeeze(nib.load(mgh).dataobj)
            cdata.append(mgh)

        # aaaand store it in the dataframe
        data = data.assign(**{concept.name: np.hstack(cdata)})

    fname.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(fname, sep=',')

    return fname


if __name__ == '__main__':
    NSDIR.mkdir(parents=True, exist_ok=True)

    # get concepts from CogAtlas and run relevant NS meta-analysess,
    database, features = fetch_ns_data(NSDIR)
    generated = run_meta_analyses(database, features, get_cogatlas_concepts(),
                                  outdir=NSDIR)

    # get parcellations that we'll use to parcellate data
    parcellations = putils.get_cammoun_schaefer(data_dir=ROIDIR)

    # get vertex-level data first
    get_ns_data(generated, fname=PARDIR / 'vertex' / 'fsaverage5.csv')

    # now do parcellations
    for name, annotations in parcellations.items():
        for scale, annot in annotations.items():
            fname = PARDIR / name / f'{scale}.csv'
            get_ns_data(generated, fname=fname, annot=annot)
