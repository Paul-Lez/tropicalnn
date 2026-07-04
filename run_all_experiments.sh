#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA="${JULIA:-julia}"

run_from_root() {
    local script="$1"
    echo "==> $script"
    (cd "$ROOT" && "$JULIA" --project="$ROOT" "$script")
}

run_in_dir() {
    local dir="$1"
    local script="$2"
    echo "==> $dir/$script"
    (cd "$ROOT/$dir" && "$JULIA" --project="$ROOT" "$script")
}

run_in_dir "visualize_linear_regions" "main.jl"
run_in_dir "effective_radius" "main.jl"

run_from_root "width_depth/main.jl"
run_from_root "rate_of_pruning/main.jl"

run_from_root "volume_dynamics/main.jl"
run_from_root "volume_dynamics/analyse.jl"

run_from_root "mnist/main.jl"
run_from_root "mnist/analyse.jl"
