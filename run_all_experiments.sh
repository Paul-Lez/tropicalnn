#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA="${JULIA:-julia}"

# Which LinearRegionsCalculationMode to use for region enumeration: "highs" (default) or "oscar".
# Change this via `TROPICALNN_MODE=oscar ./run_all_experiments.sh` or `./run_all_experiments.sh --mode=oscar`.
TROPICALNN_MODE="${TROPICALNN_MODE:-highs}"
for arg in "$@"; do
    case "$arg" in
        --mode=*) TROPICALNN_MODE="${arg#--mode=}" ;;
    esac
done
export TROPICALNN_MODE
echo "Using LP mode: $TROPICALNN_MODE"

run_from_root() {
    local script="$1"
    echo "==> $script"
    (cd "$ROOT" && "$JULIA" +1.12.2 --project="$ROOT" "$script")
}

run_in_dir() {
    local dir="$1"
    local script="$2"
    echo "==> $dir/$script"
    (cd "$ROOT/$dir" && "$JULIA" +1.12.2 --project="$ROOT" "$script")
}

run_in_dir "visualize_linear_regions" "main.jl"
run_in_dir "effective_radius" "main.jl"

run_from_root "width_depth/main.jl"
run_from_root "rate_of_pruning/main.jl"

run_from_root "volume_dynamics/main.jl"
run_from_root "volume_dynamics/analyse.jl"

run_from_root "mnist/main.jl"
run_from_root "mnist/analyse.jl"
