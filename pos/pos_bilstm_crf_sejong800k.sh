#!/bin/sh

MODE=train

INPUT_DIR=/data/pos_sejong800k/conll
OUTPUT_DIR=/data/bilstm_crf_train/pos_bilstm_crf_sejong800k_v1
NECESSARY_FILE=/data/bilstm_crf_train/pos_bilstm_crf_sejong800k_v1/vocab.pkl
TRAIN_LINES=5000000

EPOCHS=30
BATCH_SIZE=128
LEARNING_RATE=0.02
KEEP_PROB=0.65

WORD_EMBEDDING_SIZE=50
CHAR_EMBEDDING_SIZE=50

LSTM_UNITS=300
CHAR_LSTM_UNITS=300
SENTENCE_LENGTH=100
WORD_LENGTH=8

export CUDA_VISIBLE_DEVICES=2


python -u -m main \
  --mode=$MODE \
  --input_dir=$INPUT_DIR \
  --output_dir=$OUTPUT_DIR \
  --necessary_file=$NECESSARY_FILE \
  --train_lines=$TRAIN_LINES \
  --epochs=$EPOCHS \
  --batch_size=$BATCH_SIZE \
  --learning_rate=$LEARNING_RATE \
  --keep_prob=$KEEP_PROB \
  --word_embedding_size=$WORD_EMBEDDING_SIZE \
  --char_embedding_size=$CHAR_EMBEDDING_SIZE \
  --lstm_units=$LSTM_UNITS \
  --char_lstm_units=$CHAR_LSTM_UNITS \
  --sentence_length=$SENTENCE_LENGTH \
  --word_length=$WORD_LENGTH
