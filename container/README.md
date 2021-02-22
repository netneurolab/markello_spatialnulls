# container

This directory contains files used to generate a [Singularity container](https://sylabs.io/docs/) that can be used to re-run analyses without the need to install extra software (beyond, of course, Singularity).
You can download the Singularity image [on OSF](https://osf.io/za7fn/).
(It is important to note that the analyses reported in the manuscript were *not run* in the provided Singularity container; we simply provide it as a service to future researchers interested in re-running our analyses.)

- [**`gen_simg.sh`**](./gen_simg.sh): This is a helper script designed to create the Singularity image.
- [**`Singularity`**](./Singularity): The Singularity recipe generated from `gen_simg.sh` and used to build the Singularity image.

Once you've downloaded the Singularity container you _should_ (famous last words) be able to reproduce all analyses with the following command:

```bash
singularity exec --cleanenv                             \
                 --home ${PWD}                          \
                 container/markello_spatialnulls.simg   \
                 /neurodocker/startup.sh                \
                 make all
```

(Note: we don't recommend using `make all` to run the analyses because this will take an _incredibly_ long time with default settings.
Please refer to [our walkthrough](https://netneurolab.github.io/markello_spatialnulls) for guidelines and suggestions on reproducing our analyses!)
