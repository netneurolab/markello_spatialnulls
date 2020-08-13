# container

This directory contains files used to generate a [Singularity container](https://sylabs.io/docs/) that can be used to re-run analyses without the need to install extra software (beyond, of course, Singularity).
You can download the Singularity image [on OSF](https://osf.io/za7fn/).
(It is important to note that the analyses reported in the manuscript were *not run* in the provided Singularity container; we simply provide it as a service to future researchers interested in re-running our analyses.)

- [**`gen_simg.sh`**](./gen_simg.sh): This is a helper script designed to create the Singularity image.
- [**`Singularity`**](./Singularity): The Singularity recipe generated from `gen_simg.sh` and used to build the Singularity image.
