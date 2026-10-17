#!/usr/bin/env bash
set -euo pipefail

BASE="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main"
cd "$(dirname "$0")/../models"

for f in qwen2.5-0.5b-instruct-fp16.gguf \
         qwen2.5-0.5b-instruct-q8_0.gguf \
         qwen2.5-0.5b-instruct-q5_k_m.gguf \
         qwen2.5-0.5b-instruct-q4_k_m.gguf \
         qwen2.5-0.5b-instruct-q2_k.gguf; do
  if [ -f "$f" ]; then
    echo "already have $f"
    continue
  fi
  echo "downloading $f"
  curl -L -C - --fail -o "$f.part" "$BASE/$f"
  mv "$f.part" "$f"
done
ls -lh .
