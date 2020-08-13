# parspin

This directory is a tiny little Python package that we wrote to be used throughout our analyses.
These codebits aren't _necessarily_ very generalizable (though, for example, `parspin.burt` could be used in other projects!), but are re-used at various points throughout preprocessing, analysis, and results generation so they exist here rather than in any one single script.

You can check out the docstring of each module or simply read below to get an idea of what they do:

## [parspin.burt](./parspin/burt.py)

This module contains our implementation of surrogate map generation as in Burt et al., 2018, *Nat Neuro*.
The functions are annotated with relevant portions of the methods text from that manuscript that described the methodology.

## [parspin.partitions](./parspin/partitions.py)

This module contains helper code for generating the partition labels used in the "Testing partition specificity" section of the manuscript.
Since we were using multiple parcellation atlases / resolutions (Cammoun 2012 and Schaefer 2018) and several partitions (Yeo intrinsic networks and von Economo cytoarchitectonic classes) we needed some easy functions for creating + grabbing the relevant annotation files—and this module contains that code!

## [parspin.surface](./parspin/surface.py)

This module contains code for generating surface distance matrices.
This was important because we needed fast, easy-to-use code that would let us (dis)-allow travel along the medial wall that was, critically, written in pure-Python.
Much of this code has been migrated to [`BrainSMASH`](https://brainsmash.readthedocs.io/) for broader use!

## [parspin.surrogates](./parspin/surrogates.py)

Common functionality for generating surrogate brain maps for the Burt-2018 and Burt-2020 methods.
Since these methods both depend on the brain map of interest (and therefore surrogates must be generated independently for each map), we needed to make the functions parallelizable to speed up computation.

## [parspin.utils](./parspin/utils.py)

As the name suggests, this module contains simple utility functions that were moderately useful at various points of analysis.
