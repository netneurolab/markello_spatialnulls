#!/usr/bin/env bash
#
# Description:
#
#     This script is used to generate a Singularity container that can be used
#     to run all the analyses reported in our manuscript.
#
#     This script was initially written to be used on a Linux box running
#     Ubuntu 18.04. YMMV if you try it on any other system! At the very least,
#     you can extract the relevant codebits for creating the Singularity
#     container and run that (or use the pre-generated Singularity recipe in
#     the same directory as this script).
#
# Usage:
#
#     $ bash container/gen_simg.sh
#

curr_dir=$PWD && if [ ${curr_dir##*/} = "container" ]; then cd ..; fi
tag=markello_spatialnulls

# use neurodocker (<3) to make a Singularity recipe and build the Singularity
# image. this should only happen if the image doesn't already exist
if [ ! -f container/${tag}.simg ]; then
  if [ ! -f container/license.txt ]; then touch container/license.txt; fi
  singularity --quiet exec docker://repronim/neurodocker:0.7.0                \
    /usr/bin/neurodocker generate singularity                                 \
    --base ubuntu:18.04                                                       \
    --pkg-manager apt                                                         \
    --install                                                                 \
      git less nano connectome-workbench                                      \
    --freesurfer                                                              \
      version=6.0.0-min                                                       \
      license_path=container/license.txt                                      \
      exclude_paths=subjects/V1_average                                       \
    --copy ./environment.yml /opt/environment.yml                             \
    --miniconda                                                               \
      create_env=${tag}                                                       \
      yaml_file=/opt/environment.yml                                          \
    --add-to-entrypoint "source activate ${tag}"                              \
    --add-to-entrypoint "pip install -e parspin"                              \
  > container/Singularity
  sudo singularity build container/${tag}.simg container/Singularity
fi
