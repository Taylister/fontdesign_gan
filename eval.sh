#!/usr/bin/env bash
set -eu

TRAINED="2020-10-05_185356"
GAN_DIR="./result/${TRAINED}"
CSV_FILE_PATH="${GAN_DIR}/eval/information.csv"
NPY_FILE_PATH="${GAN_DIR}/eval/vectors"


python main.py --eval --gan_dir ${GAN_DIR} --csv_file ${CSV_FILE_PATH} --npy_file ${NPY_FILE_PATH}

