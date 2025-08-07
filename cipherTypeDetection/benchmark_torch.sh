#!/bin/bash

BASE_DIR=benchmark_logs
mkdir -p "$BASE_DIR"

COMMON_ARGS="--download_dataset=False \
    --plaintext_input_directory=../data/gutenberg_en \
    --rotor_input_directory=../data/rotor_ciphertexts \
    --train_dataset_size=244 \
    --batch_size=128 \
    --max_iter=1000 \
    --min_train_len=100 \
    --max_train_len=1000 \
    --min_test_len=100 \
    --max_test_len=1000 \
    --ciphers=all"

run_benchmark () {
  ARCH=$1
  for i in {1..3}; do
    RUN_DIR="$BASE_DIR/${ARCH}_run_$i"
    mkdir -p "$RUN_DIR"
    echo "Launching $ARCH run $i..."
    python train.py \
      --architecture=$ARCH \
      $COMMON_ARGS \
      --save_directory="$RUN_DIR" \
      --model_name="${ARCH}_run_$i.pth" \
      > "$RUN_DIR/${ARCH}_var_10000000_run_${i}_${DATE}.txt" \
      2> "$RUN_DIR/err_${ARCH}_var_10000000_run_${i}_${DATE}.txt"
  done
}

# FFNN
run_benchmark FFNN


# LSTM
run_benchmark LSTM
