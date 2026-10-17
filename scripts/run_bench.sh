#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
BENCH=../llama.cpp/build/bin/llama-bench
rm -f results/quant_speed.md

for q in fp16 q8_0 q5_k_m q4_k_m q2_k; do
  m="models/qwen2.5-0.5b-instruct-$q.gguf"
  echo "== $q"
  $BENCH -m "$m" -t 4 -p 128 -n 64 -r 3 -o md | tee -a results/quant_speed.md
  $BENCH -m "$m" -t 4 -p 128 -n 64 -r 3 -o csv > "results/quant_speed_$q.csv"
done
