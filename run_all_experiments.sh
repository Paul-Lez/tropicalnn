#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA="${JULIA:-julia}"
EXPERIMENT_ARGS=("$@")

run_from_root() {
    local script="$1"
    shift
    echo "==> $script"
    (cd "$ROOT" && "$JULIA" +1.12.5 --project="$ROOT" "$script" "$@")
}

run_in_dir() {
    local dir="$1"
    local script="$2"
    shift 2
    echo "==> $dir/$script"
    (cd "$ROOT/$dir" && "$JULIA" +1.12.5 --project="$ROOT" "$script" "$@")
}

run_in_dir "visualize_linear_regions" "main.jl" "${EXPERIMENT_ARGS[@]}"
run_in_dir "effective_radius" "main.jl" "${EXPERIMENT_ARGS[@]}"
run_in_dir "effective_radius" "hoffman_tables.jl" "${EXPERIMENT_ARGS[@]}"

run_from_root "width_depth/main.jl" "${EXPERIMENT_ARGS[@]}"
run_from_root "rate_of_pruning/main.jl" "${EXPERIMENT_ARGS[@]}"
run_from_root "random_rational_regions/main.jl" "${EXPERIMENT_ARGS[@]}"

run_from_root "volume_dynamics/main.jl" "${EXPERIMENT_ARGS[@]}"
run_from_root "volume_dynamics/analyse.jl"

run_from_root "mnist/main.jl"
run_from_root "mnist/analyse.jl" "${EXPERIMENT_ARGS[@]}"
