#!/usr/bin/env bash

# vLLM-Omni diffusion model benchmark for InferenceX.
#
# Benchmarks image/video generation models (e.g. Z-Image-Turbo) served by
# vllm-omni. Uses vllm-omni's diffusion_benchmark_serving.py for metrics
# (latency, throughput, SLO attainment, stage durations, etc.).
#
# Required env vars:
#   MODEL           - HuggingFace model ID or local path
#   TP              - Tensor parallelism degree
#   CONC            - Max concurrency for benchmark requests
#   RESULT_FILENAME - Output JSON filename (without directory)
#
# Optional env vars:
#   IMAGE_WIDTH           - Generated image width  (default: 1024)
#   IMAGE_HEIGHT          - Generated image height (default: 1024)
#   NUM_INFERENCE_STEPS   - Diffusion denoising steps (default: 20)
#   SEED                  - Diffusion random seed (default: 42)
#   DIFFUSION_TASK        - Task type: t2v, i2v, ti2v, ti2i, i2i, t2i (default: t2i)
#   DIFFUSION_DATASET     - Dataset: vbench, trace, random (default: random)
#   DATASET_PATH          - Optional dataset path passed to diffusion benchmark
#   PORT                  - Server port (default: 8000)
#   NUM_GPUS              - Number of GPUs for diffusion model (--num-gpus)
#   USP                   - Ulysses Sequence Parallelism degree
#   RING                  - Ring Sequence Parallelism degree
#   VAE_PATCH_PARALLEL_SIZE - VAE patch parallelism degree
#   CFG_PARALLEL_SIZE     - CFG parallel size (1 or 2)
#   NUM_FRAMES            - Number of frames for video tasks
#   FPS                   - FPS for video tasks
#   WARMUP_NUM_INFERENCE_STEPS - Warmup num_inference_steps override
#   WARMUP_CONCURRENCY    - Warmup concurrency override
#   ENABLE_NEGATIVE_PROMPT - Enable negative prompt generation for random dataset
#   RANDOM_REQUEST_CONFIG - JSON random request profile config
#   NUM_INPUT_IMAGES      - Number of synthetic input images for image-conditioned random dataset
#   GPU_MEM_UTIL          - GPU memory utilization (default: 0.90)
#   TRUST_REMOTE_CODE     - Whether to pass --trust-remote-code (default: true)

set -euo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    RESULT_FILENAME

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

nvidia-smi

# Some runners pass a pre-staged local model path, so skip `hf download`.
# Only fetch when MODEL looks like a Hugging Face repo ID.
if [[ "$MODEL" != /* ]]; then
    hf download "$MODEL"
fi

PORT=${PORT:-8000}
SERVER_LOG="${SERVER_LOG:-$PWD/server.log}"
IMAGE_WIDTH="${IMAGE_WIDTH:-1024}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-1024}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
SEED="${SEED:-42}"
DIFFUSION_TASK="${DIFFUSION_TASK:-t2i}"
DIFFUSION_DATASET="${DIFFUSION_DATASET:-random}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

DIFFUSION_BENCH_SCRIPT="$(python3 -c "
import importlib.util, os
spec = importlib.util.find_spec('vllm_omni')
if spec and spec.origin:
    pkg_dir = os.path.dirname(spec.origin)
    candidate = os.path.join(os.path.dirname(pkg_dir), 'benchmarks', 'diffusion', 'diffusion_benchmark_serving.py')
    if os.path.isfile(candidate):
        print(candidate)
" 2>/dev/null)" || true

if [[ -z "$DIFFUSION_BENCH_SCRIPT" ]]; then
    echo "ERROR: Cannot locate diffusion_benchmark_serving.py via vllm_omni package."
    exit 1
fi

export VLLM_ENGINE_READY_TIMEOUT_S=3600

PARALLEL_ARGS=(--tensor-parallel-size "$TP")

DIFFUSION_ARGS=()
if [[ -n "${NUM_GPUS:-}" ]]; then
    DIFFUSION_ARGS+=(--num-gpus "$NUM_GPUS")
fi
if [[ -n "${USP:-}" ]]; then
    DIFFUSION_ARGS+=(--usp "$USP")
fi
if [[ -n "${RING:-}" ]]; then
    DIFFUSION_ARGS+=(--ring "$RING")
fi
if [[ -n "${VAE_PATCH_PARALLEL_SIZE:-}" ]] && [ "${VAE_PATCH_PARALLEL_SIZE}" -gt 1 ]; then
    DIFFUSION_ARGS+=(--vae-patch-parallel-size "$VAE_PATCH_PARALLEL_SIZE")
fi
if [[ -n "${CFG_PARALLEL_SIZE:-}" ]] && [ "${CFG_PARALLEL_SIZE}" -gt 1 ]; then
    DIFFUSION_ARGS+=(--cfg-parallel-size "$CFG_PARALLEL_SIZE")
fi

TRUST_REMOTE_CODE_ARG=()
if [[ "${TRUST_REMOTE_CODE:-true}" = "true" ]]; then
    TRUST_REMOTE_CODE_ARG=(--trust-remote-code)
fi

GPU_METRICS_FILE="${GPU_METRICS_FILE:-$PWD/gpu_metrics.csv}"
SERVER_PID=""

cleanup() {
    local exit_code=$?
    trap - EXIT
    stop_gpu_monitor || true
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit "$exit_code"
}

trap cleanup EXIT

start_gpu_monitor --output "$GPU_METRICS_FILE"

set -x
vllm serve "$MODEL" --omni --host 0.0.0.0 --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    "${DIFFUSION_ARGS[@]}" \
    "${TRUST_REMOTE_CODE_ARG[@]}" > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

DIFFUSION_BENCH_CMD=(
    python3 "$DIFFUSION_BENCH_SCRIPT"
    --backend vllm-omni
    --model "$MODEL"
    --base-url "http://0.0.0.0:$PORT"
    --dataset "$DIFFUSION_DATASET"
    --task "$DIFFUSION_TASK"
    --width "$IMAGE_WIDTH"
    --height "$IMAGE_HEIGHT"
    --num-inference-steps "$NUM_INFERENCE_STEPS"
    --seed "$SEED"
    --num-prompts "$((CONC * 10))"
    --max-concurrency "$CONC"
    --request-rate inf
    --warmup-requests "$((CONC * 2))"
    --output-file "/workspace/${RESULT_FILENAME}.json"
)

if [[ -n "${DATASET_PATH:-}" ]]; then
    DIFFUSION_BENCH_CMD+=(--dataset-path "${DATASET_PATH:-}")
fi
if [[ -n "${NUM_FRAMES:-}" ]]; then
    DIFFUSION_BENCH_CMD+=(--num-frames "${NUM_FRAMES:-}")
fi
if [[ -n "${FPS:-}" ]]; then
    DIFFUSION_BENCH_CMD+=(--fps "${FPS:-}")
fi
if [[ -n "${WARMUP_NUM_INFERENCE_STEPS:-}" ]]; then
    DIFFUSION_BENCH_CMD+=(--warmup-num-inference-steps "${WARMUP_NUM_INFERENCE_STEPS:-}")
fi
if [[ -n "${WARMUP_CONCURRENCY:-}" ]]; then
    DIFFUSION_BENCH_CMD+=(--warmup-concurrency "${WARMUP_CONCURRENCY:-}")
fi
if [[ "${ENABLE_NEGATIVE_PROMPT:-false}" = "true" ]]; then
    DIFFUSION_BENCH_CMD+=(--enable-negative-prompt)
fi
if [[ -n "${RANDOM_REQUEST_CONFIG:-}" ]]; then
    DIFFUSION_BENCH_CMD+=(--random-request-config "${RANDOM_REQUEST_CONFIG:-}")
fi
if [[ -n "${NUM_INPUT_IMAGES:-}" ]]; then
    DIFFUSION_BENCH_CMD+=(--num-input-images "${NUM_INPUT_IMAGES:-}")
fi

set -x
"${DIFFUSION_BENCH_CMD[@]}"
set +x

RESULT_FILE="/workspace/${RESULT_FILENAME}.json"
if [[ ! -f "$RESULT_FILE" ]]; then
    echo "ERROR: Benchmark output file not found: $RESULT_FILE"
    exit 1
fi

python3 -c "
import json, os, sys

result_file = '$RESULT_FILE'
with open(result_file) as f:
    diffusion = json.load(f)

conc = os.environ.get('CONC', '0')
model = os.environ.get('MODEL', '')

adapted = {
    'max_concurrency': int(conc),
    'model_id': model,
}
adapted.update(diffusion)

with open(result_file, 'w') as f:
    json.dump(adapted, f, indent=2)
print(f'Adapted diffusion results for InferenceX pipeline: {result_file}')
"

set +x
