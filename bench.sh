#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# bench.sh — ST-EEGFormer-RS CPU vs GPU inference benchmark
#
# Usage:
#   ./bench.sh                              # benchmark small (random weights)
#   ./bench.sh small 3 10                   # 3 warmup, 10 timed runs
#   ./bench.sh small 3 10 tests/vectors/encoder_small.safetensors
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VARIANTS="${1:-small}"
WARMUP="${2:-3}"
RUNS="${3:-10}"
WEIGHTS="${4:-}"

# ─── Platform Detection ─────────────────────────────────────────────────────

OS="$(uname -s)"
ARCH="$(uname -m)"

detect_cpu() {
    case "$OS" in
        Darwin)
            sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model 2>/dev/null || echo "Unknown"
            ;;
        Linux)
            grep -m1 'model name' /proc/cpuinfo 2>/dev/null | sed 's/model name\s*:\s*//' || echo "Unknown"
            ;;
        *) echo "Unknown" ;;
    esac
}

detect_cpu_features() {
    case "$OS" in
        Darwin)
            local cores threads mem
            cores="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo '?')"
            threads="$(sysctl -n hw.logicalcpu 2>/dev/null || echo '?')"
            mem="$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
            echo "${cores}C/${threads}T, ${mem}GB RAM"
            ;;
        Linux)
            local cores mem
            cores="$(nproc --all 2>/dev/null || echo '?')"
            mem="$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || echo '?')"
            echo "${cores}C, ${mem}GB RAM"
            ;;
        *) echo "unknown" ;;
    esac
}

detect_gpu_backend() {
    case "$OS" in
        Darwin) echo "metal" ;;
        Linux)
            if command -v vulkaninfo &>/dev/null || command -v nvidia-smi &>/dev/null; then
                echo "vulkan"
            else
                echo "wgpu"
            fi
            ;;
        *) echo "wgpu" ;;
    esac
}

detect_cpu_backend_features() {
    case "$OS" in
        Darwin) echo "ndarray,blas-accelerate" ;;
        *)
            if ldconfig -p 2>/dev/null | grep -q libopenblas; then
                echo "ndarray,openblas-system"
            else
                echo "ndarray"
            fi
            ;;
    esac
}

slugify() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g' | sed 's/__*/_/g;s/^_//;s/_$//'
}

# ─── Gather info ─────────────────────────────────────────────────────────────

CPU_NAME="$(detect_cpu)"
CPU_INFO="$(detect_cpu_features)"
GPU_BACKEND="$(detect_gpu_backend)"
CPU_FEATURES="$(detect_cpu_backend_features)"
GPU_FEATURE="${GPU_BACKEND}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DATE_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"
CPU_SLUG="$(slugify "$CPU_NAME")"

RESULTS_DIR="bench_results"
RUN_ID="${TIMESTAMP}_${CPU_SLUG}"
RUN_DIR="${RESULTS_DIR}/${RUN_ID}"
mkdir -p "$RUN_DIR"

WEIGHTS_ARG=""
if [ -n "$WEIGHTS" ]; then
    WEIGHTS_ARG="--weights $WEIGHTS"
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ST-EEGFormer-RS — CPU vs GPU Benchmark                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Platform:   $OS $ARCH"
echo "  CPU:        $CPU_NAME ($CPU_INFO)"
echo "  GPU backend: $GPU_BACKEND"
echo "  Variants:   $VARIANTS"
echo "  Warmup:     $WARMUP    Runs: $RUNS"
echo "  Weights:    ${WEIGHTS:-random}"
echo "  Output:     $RUN_DIR/"
echo ""

# ─── Metadata ────────────────────────────────────────────────────────────────

cat > "$RUN_DIR/meta.json" <<EOF
{
  "timestamp": "$DATE_HUMAN",
  "os": "$OS",
  "arch": "$ARCH",
  "cpu": "$CPU_NAME",
  "cpu_info": "$CPU_INFO",
  "gpu_backend": "$GPU_BACKEND",
  "variants": "$VARIANTS",
  "warmup": $WARMUP,
  "runs": $RUNS
}
EOF

# ─── Build ────────────────────────────────────────────────────────────────────

echo "━━━ Building CPU backend (features: $CPU_FEATURES) ━━━"
cargo build --release --example benchmark --features "$CPU_FEATURES" 2>&1 | tail -3
echo ""

GPU_AVAILABLE=false
echo "━━━ Building GPU backend (features: $GPU_FEATURE) ━━━"
if cargo build --release --example benchmark --no-default-features --features "$GPU_FEATURE" 2>&1 | tail -3; then
    GPU_AVAILABLE=true
    echo ""
else
    echo "  ⚠  GPU build failed — skipping GPU benchmark"
    echo ""
fi

# ─── Run ──────────────────────────────────────────────────────────────────────

run_bench() {
    local label="$1" json_file="$2" log_file="$3"
    shift 3
    echo "━━━ Running $label benchmark ━━━"
    if "$@" > "$json_file" 2>"$log_file"; then
        if python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$json_file" 2>/dev/null; then
            echo "  ✓ $label → $json_file"
            return 0
        fi
    fi
    echo "  ⚠  $label failed (see $log_file)"
    [ -f "$log_file" ] && tail -10 "$log_file"
    return 1
}

CPU_OK=false
run_bench "CPU" "$RUN_DIR/cpu.json" "$RUN_DIR/cpu.log" \
    cargo run --release --example benchmark --features "$CPU_FEATURES" \
    -- --variants "$VARIANTS" --warmup "$WARMUP" --runs "$RUNS" $WEIGHTS_ARG --json \
    && CPU_OK=true
echo ""

GPU_OK=false
if $GPU_AVAILABLE; then
    run_bench "GPU" "$RUN_DIR/gpu.json" "$RUN_DIR/gpu.log" \
        cargo run --release --example benchmark --no-default-features --features "$GPU_FEATURE" \
        -- --variants "$VARIANTS" --warmup "$WARMUP" --runs "$RUNS" $WEIGHTS_ARG --json \
        && GPU_OK=true
    echo ""
fi

# ─── Summary ─────────────────────────────────────────────────────────────────

if ! $CPU_OK; then
    echo "  ⚠  No valid CPU results"
    [ -f "$RUN_DIR/cpu.log" ] && cat "$RUN_DIR/cpu.log"
    exit 1
fi

python3 - "$RUN_DIR" "$GPU_OK" "$CPU_NAME" "$DATE_HUMAN" <<'PYEOF'
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
gpu_ok = sys.argv[2] == "true"
cpu_name = sys.argv[3]
date_str = sys.argv[4]

with open(run_dir / "cpu.json") as f:
    cpu_data = json.load(f)

gpu_data = None
if gpu_ok and (run_dir / "gpu.json").exists():
    try:
        with open(run_dir / "gpu.json") as f:
            gpu_data = json.load(f)
    except:
        pass

print(f"\n  ╔{'═'*62}╗")
print(f"  ║  {'ST-EEGFormer-RS Benchmark Summary':^58}  ║")
print(f"  ║  {cpu_name:^58}  ║")
print(f"  ╠{'═'*62}╣")
print(f"  ║  {'Variant':<12} {'Backend':<32} {'Mean':>7} {'Min':>7}  ║")
print(f"  ╠{'─'*62}╣")

for d in cpu_data:
    v = d["variant"].capitalize()
    b = d["backend"]
    inf = d["inference"]
    print(f"  ║  {v:<12} {b:<32} {inf['mean_ms']:>6.1f}  {inf['min_ms']:>6.1f}  ║")

if gpu_data:
    for d in gpu_data:
        v = d["variant"].capitalize()
        b = d["backend"]
        inf = d["inference"]
        print(f"  ║  {v:<12} {b:<32} {inf['mean_ms']:>6.1f}  {inf['min_ms']:>6.1f}  ║")

print(f"  ╚{'═'*62}╝")

if gpu_data:
    print(f"\n  Speedup (GPU vs CPU):")
    for cpu_d, gpu_d in zip(cpu_data, gpu_data):
        v = cpu_d["variant"].capitalize()
        cpu_ms = cpu_d["inference"]["mean_ms"]
        gpu_ms = gpu_d["inference"]["mean_ms"]
        if gpu_ms > 0:
            speedup = cpu_ms / gpu_ms
            faster = "GPU" if speedup > 1 else "CPU"
            ratio = speedup if speedup > 1 else 1/speedup
            print(f"    ST-EEGFormer-{v}: {faster} is {ratio:.1f}× faster  ({cpu_ms:.1f}ms → {gpu_ms:.1f}ms)")

# Channel scaling summary
print(f"\n  Channel scaling (T=768):")
for d in cpu_data:
    v = d["variant"].capitalize()
    print(f"    {v} CPU: ", end="")
    for cs in d["channel_scaling"]:
        print(f"  {cs['channels']}ch={cs['mean_ms']:.1f}ms", end="")
    print()
if gpu_data:
    for d in gpu_data:
        v = d["variant"].capitalize()
        print(f"    {v} GPU: ", end="")
        for cs in d["channel_scaling"]:
            print(f"  {cs['channels']}ch={cs['mean_ms']:.1f}ms", end="")
        print()

# Sequence scaling summary
print(f"\n  Sequence scaling (C=22):")
for d in cpu_data:
    v = d["variant"].capitalize()
    print(f"    {v} CPU: ", end="")
    for ss in d["sequence_scaling"]:
        print(f"  {ss['samples']}={ss['mean_ms']:.1f}ms", end="")
    print()
if gpu_data:
    for d in gpu_data:
        v = d["variant"].capitalize()
        print(f"    {v} GPU: ", end="")
        for ss in d["sequence_scaling"]:
            print(f"  {ss['samples']}={ss['mean_ms']:.1f}ms", end="")
        print()

PYEOF

echo ""
echo "━━━ Results saved to $RUN_DIR/ ━━━"
ls -la "$RUN_DIR/"
echo ""
echo "Done."
