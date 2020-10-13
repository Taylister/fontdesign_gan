#!/usr/bin/env bash
set -eu

TRAINED="2020-10-05-185356"
GAN_DIR="./result/${TRAINED}"
FILE_PATH="sample"

python main.py --make_dataset --gan_dir ${GAN_DIR} --file ${FILE_PATH} --gen_num 100