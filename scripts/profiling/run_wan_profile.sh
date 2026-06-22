#!/usr/bin/env bash
# Reproducible Wan profiling wrappers. Run from repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export WAN_WEIGHTS="${WAN_WEIGHTS:-/home/mod479711/Downloads/Wan2.1-T2V-1.3B-Diffusers}"
FEATURES="${FEATURES:-flash-attn,cuda,profiling}"
PROFILE_BIN=(cargo run --release --example wan-profile --features "$FEATURES" --)

usage() {
  cat <<EOF
Usage: $0 <tool> <scenario> [extra args...]

Tools:
  baseline     Run scenario with timing + nvidia-smi snapshots
  flamegraph   cargo-flamegraph on scenario
  heaptrack    heaptrack on scenario
  tracy        Tracy capture (build with --features flash-attn,cuda,profiling,tracy)
  perf         perf record + report (sys-prof)

Scenarios (wan-profile subcommands):
  synthetic-hot-op | model-load | prefill | denoise-only | decode | full-inference

Env:
  WAN_WEIGHTS   Diffusers model directory
  FEATURES      Cargo feature list
EOF
}

tool="${1:-}"
scenario="${2:-}"
shift 2 || true

case "$tool" in
  baseline)
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
    RUST_LOG="${RUST_LOG:-candle_video::profiling=trace,candle_video=info}" \
      "${PROFILE_BIN[@]}" --weights "$WAN_WEIGHTS" "$scenario" "$@"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
    ;;
  flamegraph)
    cargo flamegraph --features "$FEATURES" --example wan-profile -- \
      --weights "$WAN_WEIGHTS" "$scenario" "$@"
    ;;
  heaptrack)
    heaptrack -o "$ROOT/target/wan-${scenario}.heaptrack.zst" \
      "${PROFILE_BIN[@]}" --weights "$WAN_WEIGHTS" "$scenario" "$@"
    echo "Report: heaptrack --analyze target/wan-${scenario}.heaptrack.zst"
    ;;
  tracy)
    FEATURES="${FEATURES},tracy"
    TRACY_NO_INVARIANT_CHECK=1 RUST_LOG=candle_video::profiling=trace \
      cargo run --release --example wan-profile --features "$FEATURES" -- \
      --weights "$WAN_WEIGHTS" "$scenario" "$@"
    echo "Open Tracy GUI and connect to localhost"
    ;;
  perf)
    perf record -g -o "$ROOT/target/wan-${scenario}.perf.data" -- \
      cargo run --release --example wan-profile --features "$FEATURES" -- \
      --weights "$WAN_WEIGHTS" "$scenario" "$@"
    perf report -i "$ROOT/target/wan-${scenario}.perf.data" | head -80
    ;;
  *)
    usage
    exit 1
    ;;
esac
