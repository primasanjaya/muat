#!/usr/bin/env bash
#
# Build the muat Docker image, auto-detecting the CUDA version to bake in.
#
# What it does (automates "Step 0"):
#   1. Looks for a GPU via nvidia-smi.
#   2. Reads the max CUDA version the driver supports.
#   3. Picks the highest PyTorch-compatible CUDA build <= that max.
#   4. Builds a GPU-capable image (which also runs on CPU-only hosts), or a
#      slim CPU-only image when no usable GPU is found.
#
# Usage:
#   ./build_docker.sh              # auto-detect
#   ./build_docker.sh cpu          # force CPU-only image
#   CUDA_VERSION=12.1 ./build_docker.sh   # force a specific CUDA build
#   TAG=muat:test ./build_docker.sh       # override image tag
#
# NOTE: the image runs wherever you `docker run` it, NOT only on this build
# host. If your run nodes differ from this one, pick the LOWEST CUDA across
# your fleet (override with CUDA_VERSION=...).
set -euo pipefail

TAG="${TAG:-muat:v0.1.27}"
PLATFORM="${PLATFORM:-linux/amd64}"

# PyTorch CUDA builds available on conda-forge, highest first.
CANDIDATES="12.4 12.1 11.8"

# Resolve the CUDA version to build with.
if [ "${1:-}" = "cpu" ]; then
    CUDA_VERSION=""
    echo "[build] CPU-only image requested."
elif [ -n "${CUDA_VERSION:-}" ]; then
    echo "[build] Using CUDA_VERSION from environment: ${CUDA_VERSION}"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    DRIVER_MAX="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
    if [ -z "${DRIVER_MAX}" ]; then
        echo "[build] GPU present but could not parse CUDA version; defaulting to 11.8."
        CUDA_VERSION="11.8"
    else
        echo "[build] Driver supports up to CUDA ${DRIVER_MAX}."
        CUDA_VERSION=""
        for c in ${CANDIDATES}; do
            # pick c if c <= DRIVER_MAX (smaller value sorts first under -V)
            if [ "$(printf '%s\n%s\n' "$c" "$DRIVER_MAX" | sort -V | head -1)" = "$c" ]; then
                CUDA_VERSION="$c"
                break
            fi
        done
        if [ -z "${CUDA_VERSION}" ]; then
            echo "[build] WARNING: driver (CUDA ${DRIVER_MAX}) is older than the lowest"
            echo "        supported PyTorch build (${CANDIDATES##* }). Building CPU-only image."
        fi
    fi
else
    echo "[build] No GPU detected (nvidia-smi unavailable). Building CPU-only image."
    CUDA_VERSION=""
fi

if [ -n "${CUDA_VERSION}" ]; then
    echo "[build] Building GPU-capable image with CUDA ${CUDA_VERSION} (tag: ${TAG})."
else
    echo "[build] Building CPU-only image (tag: ${TAG})."
fi

docker build --platform "${PLATFORM}" \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    -t "${TAG}" .

echo "[build] Done: ${TAG}"
# The image ENTRYPOINT is `muat`, so a bare `docker run <tag> python ...` would execute
# `muat python ...` and fail on an invalid subcommand. --entrypoint overrides it; the
# env's bin is already first on PATH, so plain `python` resolves inside muat-env.
echo "[build] Verify GPU at runtime (on a GPU host):"
echo "        docker run --gpus all --entrypoint python ${TAG} -c 'import torch; print(torch.cuda.is_available())'"
