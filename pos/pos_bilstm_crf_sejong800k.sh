#!/bin/sh

INPUT_DIR=/data/pos_sejong800k/conll
OUTPUT_DIR=/data/bilstm_crf_train/pos_bilstm_crf_sejong800k
NECESSARY_FILE=/data/bilstm_crf_train/pos_bilstm_crf_sejong800k/vocab.pkl
TRAIN_LINES=50

export CUDA_VISIBLE_DEVICES=2

python -m main \
  --input_dir=$INPUT_DIR \
  --output_dir=$OUTPUT_DIR \
  --necessary_file=$NECESSARY_FILE \
  --train_lines=$TRAIN_LINES
