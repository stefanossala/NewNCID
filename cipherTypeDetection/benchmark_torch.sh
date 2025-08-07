#!/bin/bash

BASE_DIR=benchmark_logs/torch
mkdir -p "$BASE_DIR"

COMMON_ARGS="--download_dataset=False --plaintext_input_directory=../data/gutenberg_en --train_dataset_size=244 --batch_size=128 --min_train_len=100 --max_train_len=1000 --min_test_len=100 --max_test_len=1000 --max_iter=1000 --epochs=1 --ciphers=all"

# Function to run the benchmark for a given architecture
run_benchmark () {
  ARCH=$1
  for i in {1..3}; do
    RUN_DIR="$BASE_DIR/${ARCH}_run_$i"
    mkdir -p "$RUN_DIR"
    echo "Launching $ARCH run $i..."
    python train.py --architecture=$ARCH $COMMON_ARGS --model_name=${ARCH}_run_$i.pth > "$RUN_DIR/train_log.txt" 2>&1
  done
}

# FFNN
run_benchmark FFNN

# LSTM
run_benchmark LSTM
