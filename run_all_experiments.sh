#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIA="${JULIA:-julia}"
EXPERIMENT_ARGS=("$@")
TROPICALNN_ROOT="$ROOT/../TropicalNN.jl"

write_metadata() {
    local git_hash
    local tropicalnn_git_hash
    local run_date

    git_hash="$(git -C "$ROOT" rev-parse HEAD)"
    tropicalnn_git_hash="$(git -C "$TROPICALNN_ROOT" rev-parse HEAD)"
    run_date="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    mkdir -p "$ROOT/outputs"
    printf '{\n  "git_hash": "%s",\n  "date": "%s",\n  "tropicalnn_git_hash": "%s"\n}\n' \
        "$git_hash" "$run_date" "$tropicalnn_git_hash" \
        > "$ROOT/outputs/metadata.json"
}

run_from_root() {
    local script="$1"
    shift
    echo "==> $script"
    (cd "$ROOT" && "$JULIA" --project="$ROOT" "$script" "$@")
}

run_in_dir() {
    local dir="$1"
    local script="$2"
    shift 2
    echo "==> $dir/$script"
    (cd "$ROOT/$dir" && "$JULIA" --project="$ROOT" "$script" "$@")
}

write_metadata

run_in_dir "visualize_linear_regions" "main.jl" "${EXPERIMENT_ARGS[@]}"
run_in_dir "effective_radius" "main.jl" "${EXPERIMENT_ARGS[@]}"
run_in_dir "effective_radius" "hoffman_tables.jl" "${EXPERIMENT_ARGS[@]}"

run_from_root "width_depth/main.jl" "${EXPERIMENT_ARGS[@]}"
run_from_root "rate_of_pruning/main.jl" "${EXPERIMENT_ARGS[@]}"
run_from_root "random_rational_regions/main.jl" "${EXPERIMENT_ARGS[@]}"

run_from_root "volume_dynamics/main.jl" "${EXPERIMENT_ARGS[@]}"
run_from_root "volume_dynamics/analyse.jl"

run_from_root "mnist/main.jl" "${EXPERIMENT_ARGS[@]}"
run_from_root "mnist/analyse.jl" "${EXPERIMENT_ARGS[@]}"
