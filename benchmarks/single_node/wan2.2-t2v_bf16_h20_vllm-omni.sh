#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/zimage-turbo_bf16_h20_vllm-omni.sh"

MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}" \
DIFFUSION_TASK="${DIFFUSION_TASK:-t2v}" \
IMAGE_WIDTH="${IMAGE_WIDTH:-832}" \
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}" \
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}" \
NUM_FRAMES="${NUM_FRAMES:-33}" \
FPS="${FPS:-16}" \
"$BASE_SCRIPT" "$@"
