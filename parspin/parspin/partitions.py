# -*- coding: utf-8 -*-
"""
Code for aiding in partition specificity analyses
"""

from collections import namedtuple
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats

from netneurotools import datasets as nndata, freesurfer as nnsurf

YEO_CODES = {
    'visual': 1,
    'somatomotor': 2,
    'dorsal attention': 3,
    'ventral attention': 4,
    'limbic': 5,
    'frontoparietal': 6,
    'default mode': 7
}
VEK_CODES = {
    'association cortex 1': 1,
    'association cortex 2': 2,
    'insular cortex': 3,
    'limbic regions': 4,
    'primary motor cortex': 5,
    'primary sensory cortex': 6,
    'primary/secondary sensory': 7,
}

# helpful little namedtuple for working with the RSN affiliations
NETWORKS = namedtuple('Networks', ('vertices', 'parcels'))


def get_schaefer2018_yeo(scale, data_dir=None):
    """
    Returns Yeo RSN affiliations for Schaefer parcellation

    Parameters
    ----------
    scale : str
        Scale of Schaefer et al., 2018 to use
    data_dir : str or os.PathLike
        Directory where parcellation should be downloaded to (or exists, if
        already downloaded). Default: None

    Returns
    -------
    labels : (2,) namedtuple
        Where the first entry ('vertices') is the vertex-level RSN affiliations
        and the second entry ('parcels') is the parcel-level RSN affiliations
    """

    # substring in schaefer parcel names indiciating yeo network affiliation
    schaefer_yeo = dict(
        Vis='visual',
        SomMot='somatomotor',
        DorsAttn='dorsal attention',
        SalVentAttn='ventral attention',
        Limbic='limbic',
        Cont='frontoparietal',
        Default='default mode'
    )

    # get requested annotation files
    schaefer = nndata.fetch_schaefer2018('fsaverage5',
                                         data_dir=data_dir)[scale]

    network_labels, parcel_labels = [], []
    for hemi in ('lh', 'rh'):
        # read in annotation file for given hemisphere
        annot = getattr(schaefer, hemi)
        labels, ctab, names = nib.freesurfer.read_annot(annot)
        names = [m.decode() for m in names]

        # create empty arrays for vertex- and parcel-level affiliations
        networks = np.zeros_like(labels)
        parcels = np.zeros(len(names), dtype=int)
        for n, parcel in enumerate(names):
            for abbrev, net in schaefer_yeo.items():
                # check which network this parcel belongs to by looking for the
                # network substring in the parcel name and assign accordingly
                if abbrev in parcel:
                    parcels[n] = YEO_CODES[net]
                    networks[labels == n] = YEO_CODES[net]

        # store network affiliations for this hemisphere
        network_labels.append(networks)
        parcel_labels.append(parcels)

    return NETWORKS(np.hstack(network_labels), np.hstack(parcel_labels))


def get_cammoun2012_yeo(scale, data_dir=None):
    """
    Returns Yeo RSN affiliations for Cammoun parcellation

    Parameters
    ----------
    scale : str
        Scale of Cammoun et al., 2012 to use
    data_dir : str or os.PathLike
        Directory where parcellation should be downloaded to (or exists, if
        already downloaded). Default: None

    Returns
    -------
    labels : (2,) namedtuple
        Where the first entry ('vertices') is the vertex-level RSN affiliations
        and the second entry ('parcels') is the parcel-level RSN affiliations
    """

    # get requested annotation files
    cammoun = nndata.fetch_cammoun2012('fsaverage5', data_dir=data_dir)[scale]

    # we also need to load in the CSV file with info about the parcellation.
    # unlike the Schaefer et al parcellation the labels in our annotation file
    # provide no information about the network affiliation
    info = pd.read_csv(nndata.fetch_cammoun2012(data_dir=data_dir)['info'])
    info = info.query(f'scale == "{scale}"')

    network_labels, parcel_labels = [], []
    for hemi in ('lh', 'rh'):
        # query dataframe for information for current hemisphere
        cinfo = info.query(f'hemisphere == "{hemi[0].capitalize()}"')

        # read in annotation file for given hemisphere
        annot = getattr(cammoun, hemi)
        labels, ctab, names = nib.freesurfer.read_annot(annot)
        names = [m.decode() for m in names]

        # create empty arrays for vertex- and parcel-level affiliations
        networks = np.zeros_like(labels)
        parcels = np.zeros(len(names), dtype=int)
        for n, parcel in enumerate(names):
            # these should be 'background' parcels (unknown / corpuscallosum)
            if parcel not in list(info['label']):
                continue
            # get the yeo affiliation from the dataframe and assign accordingly
            net = np.squeeze(cinfo.query(f'label == "{parcel}"')['yeo_7'])
            parcels[n] = YEO_CODES[net]
            networks[labels == n] = YEO_CODES[net]

        # store network affiliations for this hemisphere
        network_labels.append(networks)
        parcel_labels.append(parcels)

    return NETWORKS(np.hstack(network_labels), np.hstack(parcel_labels))


def get_vertex_yeo(*args, **kwargs):
    """
    Returns Yeo RSN affiliations for fsaverage5 surface

    Returns
    -------
    labels : (2,) namedtuple
        Where the first entry ('vertices') is the vertex-level RSN affiliations
        and the second entry ('parcels') is the parcel-level RSN affiliations
    """

    # check for FreeSurfer installation and get path to fsaverage5 label dir
    subjdir = Path(nnsurf.check_fs_subjid('fsaverage5')[-1]).resolve()
    labeldir = subjdir / 'fsaverage5' / 'label'

    # load Yeo2011 network affiliations and store in expected output
    network_labels, parcel_labels = [], []
    for hemi in ('lh', 'rh'):
        annot = labeldir / f'{hemi}.Yeo2011_7Networks_N1000.annot'
        labels, ctab, names = nib.freesurfer.read_annot(annot)
        network_labels.append(labels)
        parcel_labels.append(range(len(names)))

    return NETWORKS(np.hstack(network_labels), np.hstack(parcel_labels))


def _apply_vek_prob(data_dir=None):
    """
    Applies probabilistic von Economo & Koskinas FreeSurfer classifier

    Uses `fsaverage5` surface; requires FreeSurfer installation.

    Parameters
    ----------
    data_dir : str or os.PathLike, optional
        Path where probabilistic von Economo & Koskinas classifier should be
        downloaded. Also determines where generated annotation files are
        saved.

    Returns
    -------
    surface : (2,) namedtuple
        Where the first entry ('lh') is the left hemisphere annotation file and
        the second entry ('rh') is the right hemisphere annotation file
    """

    vek = nndata.fetch_voneconomo(data_dir=data_dir)

    annots = []
    for hemi in ('lh', 'rh'):
        gcs = Path(getattr(vek['gcs'], hemi))
        ctab = Path(getattr(vek['ctab'], hemi))
        annot = (
            gcs.parent / 'atl-vonEconomoKoskinas_space-fsaverage5_hemi-{}'
                         '_deterministic.annot'.format(hemi[0].capitalize())
        )
        annot = nnsurf.apply_prob_atlas('fsaverage5', str(gcs), hemi,
                                        ctab=str(ctab), annot=str(annot))
        annots.append(annot)

    return vek['gcs'].__class__(*annots)


def _convert_vek_to_classes(annot, data_dir=None):
    """
    Converts von Economo regional designations to cytoarchitectonic classes

    Parameters
    ----------
    annot : str or os.PathLike
        Path to von Economo FreeSurfer annotation file

    Returns
    -------
    classes : numpy.ndarray
        Class designations for each vertex of provided `annot`
    """

    vek = nndata.fetch_voneconomo(data_dir=data_dir)
    info = pd.read_csv(vek['info'], dtype={'class': 'category'})
    if isinstance(annot, str):
        annot = nib.freesurfer.read_annot(annot)[0]
    return np.asarray(info['class'].cat.codes)[annot] + 1


def _parcellate_vek_classes(parc, vek_annots):
    """
    Uses `parc` to parcellate `vek_annots` via winner-take-all approach

    Parameters
    ----------
    parc : (2,) namedtuple
        With entries ('lh', 'rh') denoting the annotation files for the
        specified parcellation
    vek_annots : (2,) namedtuple
        With entries ('lh', 'rh') denoting the annotation files for the von
        Economo classes

    Returns
    -------
    labels : (2,) namedtuple
        Where the first entry ('vertices') is the vertex-level classes and the
        second entry ('parcels') is the parcel-level classes for the specified
        parcellation
    """

    vertex_labels, parcel_labels = [], []
    for hemi in ('lh', 'rh'):
        pl, _, pn = nib.freesurfer.read_annot(getattr(parc, hemi))
        vl, *_ = nib.freesurfer.read_annot(getattr(vek_annots, hemi))
        vc = _convert_vek_to_classes(vl)
        labs = ndimage.labeled_comprehension(vc, pl, index=range(len(pn)),
                                             func=lambda x: stats.mode(x)[0],
                                             out_dtype=int, default=-1)

        # store network affiliations for this hemisphere
        vertex_labels.append(labs[pl])
        parcel_labels.append(labs)

    return NETWORKS(np.hstack(vertex_labels), np.hstack(parcel_labels))


def get_schaefer2018_vek(scale, data_dir=None):
    """
    Returns von Economo cytoarchitectonic classes for Schaefer parcellation

    Parameters
    ----------
    scale : str
        Scale of Schaefer et al., 2018 to use
    data_dir : str or os.PathLike
        Directory where parcellation should be downloaded to (or exists, if
        already downloaded). Default: None

    Returns
    -------
    labels : (2,) namedtuple
        Where the first entry ('vertices') is the vertex-level classes and the
        second entry ('parcels') is the parcel-level classes
    """

    schaefer = nndata.fetch_schaefer2018('fsaverage5',
                                         data_dir=data_dir)[scale]
    vek_annots = _apply_vek_prob(data_dir=data_dir)
    return _parcellate_vek_classes(schaefer, vek_annots)


def get_cammoun2012_vek(scale, data_dir=None):
    """
    Returns von Economo cytoarchitectonic classes for Cammoun parcellation

    Parameters
    ----------
    scale : str
        Scale of Cammoun et al., 2012 to use
    data_dir : str or os.PathLike
        Directory where parcellation should be downloaded to (or exists, if
        already downloaded). Default: None

    Returns
    -------
    labels : (2,) namedtuple
        Where the first entry ('vertices') is the vertex-level classes and the
        second entry ('parcels') is the parcel-level classes
    """

    cammoun = nndata.fetch_cammoun2012('fsaverage5', data_dir=data_dir)[scale]
    vek_annots = _apply_vek_prob(data_dir=data_dir)
    return _parcellate_vek_classes(cammoun, vek_annots)


def get_vertex_vek(*args, data_dir=None, **kwargs):
    """
    Returns von Economo cytoarchitectonic classes for fsaverage5 surface

    Returns
    -------
    labels : (2,) namedtuple
        Where the first entry ('vertices') is the vertex-level classes and the
        second entry ('parcels') is the unique classes
    """

    annots = _apply_vek_prob(data_dir=data_dir)
    vertex_labels, parcel_labels = [], []
    for hemi in ('lh', 'rh'):
        vc = _convert_vek_to_classes(getattr(annots, hemi), data_dir=data_dir)
        vertex_labels.append(vc)
        parcel_labels.append(np.unique(vc))
    return NETWORKS(np.hstack(vertex_labels), np.hstack(parcel_labels))


YEO_AFILLIATION = {
    'vertex': get_vertex_yeo,
    'atl-cammoun2012': get_cammoun2012_yeo,
    'atl-schaefer2018': get_schaefer2018_yeo
}

VEK_AFFILIATION = {
    'vertex': get_vertex_vek,
    'atl-cammoun2012': get_cammoun2012_vek,
    'atl-schaefer2018': get_schaefer2018_vek
}

NET_OPTIONS = {
    'yeo': YEO_AFILLIATION,
    'vek': VEK_AFFILIATION
}

NET_CODES = {
    'yeo': YEO_CODES,
    'vek': VEK_CODES
}
