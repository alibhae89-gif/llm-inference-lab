#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PPL=../llama.cpp/build/bin/llama-perplexity
TEXT=data/wikitext-2-raw/wiki.test.raw
rm -f results/quant_ppl.txt

for q in fp16 q8_0 q5_k_m q4_k_m q2_k; do
  m="models/qwen2.5-0.5b-instruct-$q.gguf"
  echo "== $q"
  $PPL -m "$m" -f "$TEXT" -t 4 -c 512 --chunks 20 > "results/ppl_$q.log" 2>&1
  echo "$q: $(grep 'Final estimate' "results/ppl_$q.log")" | tee -a results/quant_ppl.txt
done
