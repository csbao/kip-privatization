#! /bin/bash
#
# Setup conda environment and data.

set -euo pipefail

THIS_DIR=$(dirname $0)
ROOT_DIR=$(dirname $THIS_DIR)

env_name=kip
conda_root=$(dirname $(dirname $(which conda)))
. "$conda_root/etc/profile.d/conda.sh"
conda env remove -n $env_name
conda create -y -n ${env_name} python=3.11 pip git-lfs

conda activate ${env_name}
pip install -r ${ROOT_DIR}/requirements.txt
python -m spacy download en_core_web_sm

models_dir=$ROOT_DIR/models
mkdir -p $models_dir



base_path=$models_dir/dipper-large
ckpt_path=$models_dir/dipper_cd_v130.bin
if [[ ! -e $base_path ]]; then
    echo "$base_path must be pre-fetched manually before running this script"
    echo "please download the models from X [TODO: add link]"
    exit 1
fi

if [[ ! -e $ckpt_path ]]; then
    echo "$ckpt_path must be pre-fetched manually before running this script"
    echo "please download the models from X [TODO: add link]"
    exit 1
fi



