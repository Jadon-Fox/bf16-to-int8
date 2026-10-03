#!/usr/bin/env bash
# Build GPU INT8 block-quant for the pin converter. Not train.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
nvcc -O2 -std=c++17 -arch=sm_86 -shared -Xcompiler -fPIC \
  -o "$ROOT/cuda/libquant_i8.so" "$ROOT/cuda/quant_i8.cu" -lcudart
ls -l "$ROOT/cuda/libquant_i8.so"
echo BUILD_QUANT_I8_OK sm_86
