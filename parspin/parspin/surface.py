# -*- coding: utf-8 -*-
"""
Various surface helper functions
"""

import os
from pathlib import Path
import tempfile
from subprocess import run

from joblib import Parallel, delayed
import nibabel as nib
import numpy as np
from scipy import ndimage, sparse

from netneurotools.surface import make_surf_graph
from parspin.utils import trange

FS_SUFFIXES = [
    '.orig', '.white', '.smoothwm', '.pial', '.inflated', '.sphere'
]
CIFTITOGIFTI = 'wb_command -cifti-separate {cifti} COLUMN -label {hemi} {gii}'


def _decode(x):
    """ Decodes `x` if it has the `.decode()` method
    """
    if hasattr(x, 'decode'):
        return x.decode()
    return x


def pathify(path):
    """
    Convenience function for coercing a potential pathlike to a Path object

    Parameter
    ---------
    path : str or os.PathLike
        Path to be checked for coercion to pathlib.Path object

    Returns
    -------
    path : pathlib.Path
    """

    if isinstance(path, (str, os.PathLike)):
        path = Path(path)
    return path


def label_to_gii(label, surf, out=None):
    """
    Converts FreeSurfer-style label file to gifti format

    Data array in output gifti file will have values = 1 where `label`

    Parameters
    ----------
    label : str or os.PathLike
        Path to FreeSurfer label file
    surf : str or os.PathLike
        Path to gifti surface file (tp which `label` refers)
    out : str or os.PathLike
        Path to desired output file location. If not specified will use same
        directory as `annot` and replace file extension with `.dlabel.gii`

    Returns
    -------
    gifti : pathlib.Path
        Path to generated gifti file
    """

    label = pathify(label)
    if out is None:
        out = label.with_suffix('.label.gii')

    # boolean label with 1 = medial, 0 = not medial
    labels = np.zeros(len(nib.load(surf).agg_data()[0]),
                      dtype='int32')
    labels[nib.freesurfer.read_label(label)] = 1

    darray = nib.gifti.GiftiDataArray(labels, 'NIFTI_INTENT_LABEL',
                                      datatype='NIFTI_TYPE_INT32')
    gii = nib.gifti.GiftiImage(darrays=[darray])

    nib.save(gii, out)

    return out


def annot_to_gii(annot, out=None):
    """
    Converts FreeSurfer-style annotation file to gifti format

    Parameters
    ----------
    annot : str or os.PathLike
        Path to FreeSurfer annotation file
    out : str or os.PathLike
        Path to desired output file location. If not specified will use same
        directory as `annot` and replace file extension with `.dlabel.gii`

    Returns
    -------
    gifti : pathlib.Path
        Path to generated gifti file
    """

    # path handling
    annot = pathify(annot)
    if out is None:
        out = annot.with_suffix('.label.gii')

    # convert annotation to gifti internally, with nibabel
    labels, ctab, names = nib.freesurfer.read_annot(annot)
    darray = nib.gifti.GiftiDataArray(labels, 'NIFTI_INTENT_LABEL',
                                      datatype='NIFTI_TYPE_INT32')
    # create label table used ctab and names from annotation file
    labtab = nib.gifti.GiftiLabelTable()
    for n, (r, g, b, a) in enumerate(ctab[:, :-1]):
        lab = nib.gifti.GiftiLabel(n, r, g, b, a)
        lab.label = _decode(names[n])
        labtab.labels.append(lab)
    gii = nib.gifti.GiftiImage(labeltable=labtab, darrays=[darray])

    nib.save(gii, out)

    return out


def dlabel_to_gii(dlabel, out=None, use_wb=False):
    """
    Converts CIFTI surface file to gifti format

    Parameters
    ----------
    dlabel : str or os.PathLike
        Path to CIFTI dlabel file
    out : str or os.PathLike
        Path to desired output file location. If not specified will use same
        directory as `surf` and replace file extension with `.surf.gii`

    Returns
    -------
    gifti : pathlib.Path
        Path to generated gifti file
    """

    dlabel = pathify(dlabel)
    if out is None:
        out = dlabel.with_suffix('.label.gii')  # this doesn't remove .dlabel


def surf_to_gii(surf, out=None):
    """
    Converts FreeSurfer-style surface file to gifti format

    Parameters
    ----------
    surf : str or os.PathLike
        Path to FreeSurfer surface file
    out : str or os.PathLike
        Path to desired output file location. If not specified will use same
        directory as `surf` and replace file extension with `.surf.gii`

    Returns
    -------
    gifti : pathlib.Path
        Path to generated gifti file
    """

    # path handling
    surf = pathify(surf)
    if out is None:
        out = surf.with_suffix('.surf.gii')

    vertices, faces = nib.freesurfer.read_geometry(surf)
    vertices = nib.gifti.GiftiDataArray(vertices, 'NIFTI_INTENT_POINTSET',
                                        datatype='NIFTI_TYPE_FLOAT32')
    faces = nib.gifti.GiftiDataArray(faces, 'NIFTI_INTENT_TRIANGLE',
                                     datatype='NIFTI_TYPE_INT32')
    gii = nib.gifti.GiftiImage(darrays=[vertices, faces])

    nib.save(gii, out)

    return out


def _labels_to_gii(labels, surf):
    labels, remove_labels = pathify(labels), False

    if labels is None:
        return labels, remove_labels

    if labels.suffix == '.annot':
        labels = annot_to_gii(labels, out=tempfile.mkstemp('.label.gii')[1])
        remove_labels = True
    elif labels.suffix == '.dlabel.nii':
        labels = dlabel_to_gii(labels, out=tempfile.mkstemp('.label.gii')[1])
        remove_labels = True
    elif labels.suffix == '.label':
        labels = label_to_gii(labels, surf,
                              out=tempfile.mkstemp('.label.gii')[1])
        remove_labels = True

    return pathify(labels), remove_labels


def _surf_to_gii(surf):
    surf, remove_surf = pathify(surf), False

    if surf is None:
        return surf, remove_surf

    if surf.suffix in FS_SUFFIXES:
        surf = surf_to_gii(surf, out=tempfile.mkstemp('.surf.gii')[1])
        remove_surf = True

    return pathify(surf), remove_surf


def _get_workbench_distance(vertex, surf, labels=None):
    """
    Gets surface distance of `vertex` to all other vertices in `surf`

    Parameters
    ----------
    vertex : int
        Index of vertex for which to calculate surface distance
    surf : str or os.PathLike
        Path to surface file on which to calculate distance
    labels : array_like, optional
        Labels indicating parcel to which each vertex belongs. If provided,
        distances will be averaged within distinct labels

    Returns
    -------
    dist : (N,) numpy.ndarray
        Distance of `vertex` to all other vertices in `graph` (or to all
        parcels in `labels`, if provided)
    """

    distcmd = 'wb_command -surface-geodesic-distance {surf} {vertex} {out}'

    # run the geodesic distance command with wb_command
    with tempfile.NamedTemporaryFile(suffix='.func.gii') as out:
        run(distcmd.format(surf=surf, vertex=vertex, out=out.name),
            shell=True, check=True, universal_newlines=True)
        dist = nib.load(out.name).agg_data()

    if labels is not None:
        dist = ndimage.mean(input=np.delete(dist, vertex),
                            labels=np.delete(labels, vertex),
                            index=np.unique(labels))

    return dist.astype(np.float32)


def _get_graph_distance(vertex, graph, labels=None):
    """
    Gets surface distance of `vertex` to all other vertices in `graph`

    Parameters
    ----------
    vertex : int
        Index of vertex for which to calculate surface distance
    graph : array_like
        Graph along which to calculate shortest path distances
    labels : array_like, optional
        Labels indicating parcel to which each vertex belongs. If provided,
        distances will be averaged within distinct labels

    Returns
    -------
    dist : (N,) numpy.ndarray
        Distance of `vertex` to all other vertices in `graph` (or to all
        parcels in `labels`, if provided)
    """

    # this involves an up-cast to float64; we're gonne get some rounding diff
    # here when compared to the wb_command subprocess call
    dist = sparse.csgraph.dijkstra(graph, directed=False, indices=[vertex])

    if labels is not None:
        dist = ndimage.mean(input=np.delete(dist, vertex),
                            labels=np.delete(labels, vertex),
                            index=np.unique(labels))

    return dist.astype(np.float32)


def get_surface_distance(surf, dlabel=None, medial=None, medial_labels=None,
                         drop_labels=None, use_wb=False, n_proc=1,
                         verbose=False):
    """
    Calculates surface distance for vertices in `surf`

    Parameters
    ----------
    surf : str or os.PathLike
        Path to surface file on which to calculate distance
    dlabel : str or os.PathLike, optional
        Path to file with parcel labels for provided `surf`. If provided will
        calculate parcel-parcel distances instead of vertex distances. Default:
        None
    medial : str or os.PathLike, optional
        Path to file containing labels for vertices corresponding to medial
        wall. If provided (and `use_wb=False`), will disallow calculation of
        surface distance along the medial wall. Default: None
    medial_labels : list of str, optional
        List of parcel names that comprise the medial wall and through which
        travel should be disallowed (if `dlabel` provided and `use_wb=False`).
        Will supersede `medial` if both are provided. Default: None
    drop_labels : list of str, optional
        List of parcel names that should be dropped from the final distance
        matrix (if `dlabel` is provided). If not specified, will ignore all
        parcels commonly used to reference the medial wall (e.g., 'unknown',
        'corpuscallosum', '???', 'Background+FreeSurfer_Defined_Medial_Wall').
        Default: None
    use_wb : bool, optional
        Whether to use calls to `wb_command -surface-geodesic-distance` for
        computation of the distance matrix; this will involve significant disk
        I/O. If False, all computations will be done in memory using the
        :func:`scipy.sparse.csgraph.dijkstra` function. Default: False
    n_proc : int, optional
        Number of processors to use for parallelizing distance calculation. If
        negative, will use max available processors plus 1 minus the specified
        number. Default: 1 (no parallelization)
    verbose : bool, optional
        Whether to print progress bar while distances are calculated. Default:
        True

    Returns
    -------
    distance : (N, N) numpy.ndarray
        Surface distance between vertices/parcels on `surf`

    Notes
    -----
    The distance matrix computed with `use_wb=False` will have slightly lower
    values than when `use_wb=True` due to known estimation errors. These will
    be fixed at a later date.
    """

    if drop_labels is None:
        drop_labels = [
            'unknown', 'corpuscallosum', '???',
            'Background+FreeSurfer_Defined_Medial_Wall'
        ]
    if medial_labels is None:
        medial_labels = []

    # convert to paths, if necessary
    surf, dlabel, medial = pathify(surf), pathify(dlabel), pathify(medial)

    # wb_command requires gifti files so convert if we receive e.g., a FS file
    # also return a "remove" flag that will be used to delete the temporary
    # gifti file at the end of this process
    surf, remove_surf = _surf_to_gii(surf)
    n_vert = len(nib.load(surf).agg_data()[0])

    # check if dlabel / medial wall files were provided
    labels, mask = None, np.zeros(n_vert, dtype=bool)
    dlabel, remove_dlabel = _labels_to_gii(dlabel, surf)
    medial, remove_medial = _labels_to_gii(medial, surf)

    # get data from dlabel / medial wall files if they provided
    if dlabel is not None:
        labels = nib.load(dlabel).agg_data()
    if medial is not None:
        mask = nib.load(medial).agg_data().astype(bool)

    # determine which parcels should be ignored (if they exist)
    delete, uniq_labels = [], np.unique(labels)
    if (len(drop_labels) > 0 or len(medial_labels) > 0) and labels is not None:
        # get vertex labels
        n_labels = len(uniq_labels)

        # get parcel labels and reverse dict to (name : label)
        table = nib.load(dlabel).labeltable.get_labels_as_dict()
        table = {v: k for k, v in table.items()}

        # generate dict mapping label to array indices (since labels don't
        # necessarily start at 0 / aren't contiguous)
        idx = dict(zip(uniq_labels, np.arange(n_labels)))

        # get indices of parcel distance matrix to be deleted
        for lab in set(table) & set(drop_labels):
            lab = table.get(lab)
            delete.append(idx.get(lab))

        for lab in set(table) & set(medial_labels):
            lab = table.get(lab)
            mask[labels == lab] = True

    # calculate distance from each vertex to all other parcels
    parallel = Parallel(n_jobs=n_proc, max_nbytes=None)
    if use_wb:
        parfunc = delayed(_get_workbench_distance)
        graph = surf
    else:
        parfunc = delayed(_get_graph_distance)
        graph = make_surf_graph(*nib.load(surf).agg_data(), mask=mask)
    bar = trange(n_vert, verbose=verbose, desc='Calculating distances')
    dist = np.row_stack(parallel(parfunc(n, graph, labels) for n in bar))

    # average distance for all vertices within a parcel + set diagonal to 0
    if labels is not None:
        dist = np.row_stack([
            dist[labels == lab].mean(axis=0) for lab in uniq_labels
        ])
        dist[np.diag_indices_from(dist)] = 0

    # remove distances for parcels that we aren't interested in
    if len(delete) > 0:
        for axis in range(2):
            dist = np.delete(dist, delete, axis=axis)

    # if we created gifti files then remove them
    if remove_surf:
        surf.unlink()
    if remove_dlabel:
        dlabel.unlink()
    if remove_medial:
        medial.unlink()

    return dist
