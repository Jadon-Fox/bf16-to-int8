#!/usr/bin/env bash
# Build the Mojo INT8 block-quant extension (twin of cuda/build_quant_i8.sh).
# Needs `pip install mojo max` (Mojo 1.1 / MAX 26.6). No GPU needed to build:
# --target-accelerator cross-compiles the kernels for the RTX 3060 (sm_86).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCH="${MOJO_ACCEL:-sm_86}"
mojo build --emit shared-lib --target-accelerator "$ARCH" \
  -o "$ROOT/mojo/quant_i8_mojo.so" "$ROOT/mojo/quant_i8.mojo"
ls -l "$ROOT/mojo/quant_i8_mojo.so"
echo "BUILD_QUANT_I8_MOJO_OK $ARCH"
