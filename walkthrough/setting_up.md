# Set-up and installation

## Required software

Reproducing these analyses require the following software (links go to installation instructions for each dependency):

- [Git](https://git-scm.com/),
- [Python 3.6+](https://docs.conda.io/en/latest/miniconda.html),
- [FreeSurfer v6.0.0](https://surfer.nmr.mgh.harvard.edu/fswiki/rel6downloads), and
- [Connectome Workbench](https://www.humanconnectome.org/software/get-connectome-workbench).

Alternatively, you can opt to just use:

- [Singularity](https://sylabs.io/guides/3.6/user-guide/quick_start.html)

## Getting the repository with `git`

First, you'll need a copy of all the data, code, and whatnot in the repository.
You can make a copy by running the following command:

```bash
git clone https://github.com/netneurolab/markello_spatialnulls
cd markello_spatialnulls/
```

## Python dependencies

It is recommended that you create a new Python environment to install all the dependencies for the analyses.
If you'd prefer to install all the dependencies in your current Python environment you can do that, but no guarantees that things will work without issue!
(No guarantee things will work without issue even if you use environments and containers, but we're trying!)

### Using `conda` (recommended)

If you are using [`conda`](https://docs.conda.io/en/latest/miniconda.html) you can create a new environment and install all the required Python dependencies with the following command:

```bash
conda env create -f environment.yml
conda activate markello_spatialnulls
```

Alternatively you can add all the dependencies to your current environment with:

```bash
conda env update -f environment.yml
```

### Using `pip`

If you are using `pip` you can install all the dependencies into your current environment with:

```bash
pip install -r requirements.txt
```

### Internal libraries

We've written a small internal package for this project that contains some processing code for the analyses in the manuscript.
Once you've created your Python environment (or installed the dependencies as described above), you can install this package from the root of the repository with the following commands:

```bash
pip install parspin
```

## FreeSurfer and Connectome Workbench

Some of the data processing in our manuscript relies on [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/rel6downloads) and [Connectome Workbench](https://www.humanconnectome.org/software/get-connectome-workbench).
We recommend you follow the installation guides on the relevant website for each software package because procedures will vary dramatically based on OS.

Check that you have the necessary commands available by typing the following into the terminal:

```bash
mri_vol2surf --help
wb_command -cifti-separate
```

(You should see a bunch of help text printed separately for each command!)

## Singularity

If you'd prefer to not install the additional dependencies of Python, FreeSurfer, and the Connectome Workbench, you can instead opt to use [Singularity](https://sylabs.io/docs/).
We provide a Singularity image that can be used to run all the primary analyses [via OSF](https://osf.io/za7fn/).
(It is important to note that the analyses reported in the manuscript were *not run* in the provided Singularity container; we provide it as a service to future researchers interested in re-running our analyses.)
