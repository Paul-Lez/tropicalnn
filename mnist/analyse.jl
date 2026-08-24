include(joinpath(@__DIR__, "..", "experiment_setup.jl"))
const EXPERIMENT_RUNTIME = setup_experiment!()

using CSV
using DataFrames
import Flux
using JLD2
using TropicalNN

include(joinpath(@__DIR__, "models.jl"))
using .MNISTModels

const REGION_MODE = highs_mode(EXPERIMENT_RUNTIME)
const WORKER_IDS = tropical_workers(EXPERIMENT_RUNTIME)
const OUTPUT_DIR = joinpath("outputs", "mnist")
const RESULTS_PATH = joinpath(OUTPUT_DIR, "linear_regions.csv")

function empty_results()
    return DataFrame(
        Model = String[],
        Activation = String[],
        Architecture = String[],
        HiddenLayers = Int[],
        Width = Int[],
        Pieces = Union{Missing, Int}[],
        NumRegions = Int[],
        TimeSeconds = Float64[],
        Algorithm = String[],
        Encoding = String[],
    )
end

function load_results()
    isfile(RESULTS_PATH) || return empty_results()
    results = load_typed_csv(RESULTS_PATH, empty_results())
    all(results.Algorithm .== "HiGHS") || throw(ArgumentError(
        "$RESULTS_PATH contains results from an algorithm other than HiGHS"
    ))
    all(results.Encoding .== "Float64") || throw(ArgumentError(
        "$RESULTS_PATH contains results with an encoding other than Float64"
    ))
    return results
end

function write_results(results)
    temporary_path = RESULTS_PATH * ".tmp"
    CSV.write(temporary_path, results)
    mv(temporary_path, RESULTS_PATH; force = true)
end

function save_analysis(path; kwargs...)
    temporary_path = path * ".tmp"
    jldsave(temporary_path; kwargs...)
    mv(temporary_path, path; force = true)
end

function run_analysis()
    results = load_results()
    analysis_dir = joinpath(OUTPUT_DIR, "analysis")
    mkpath(analysis_dir)

    for spec in experiment_specs()
        architecture = join(vcat(28^2, spec.widths, 10), ":")
        path = model_path(OUTPUT_DIR, spec)
        isfile(path) || throw(ArgumentError("missing trained model: $path; run mnist/main.jl first"))
        analysis_path = joinpath(analysis_dir, "$(spec.id).jld2")

        matching_rows = findall(==(spec.id), results.Model)
        length(matching_rows) <= 1 || throw(ArgumentError(
            "$RESULTS_PATH contains duplicate rows for $(spec.id)"
        ))
        if !isempty(matching_rows)
            row_index = only(matching_rows)
            row = results[row_index, :]
            expected_pieces = spec.activation == :maxout ? spec.pieces : missing
            row_matches_spec = row.Activation == string(spec.activation) &&
                row.Architecture == architecture &&
                row.HiddenLayers == length(spec.widths) &&
                row.Width == first(spec.widths) &&
                isequal(row.Pieces, expected_pieces)
            row_matches_spec || throw(ArgumentError(
                "$RESULTS_PATH contains incompatible metadata for $(spec.id)"
            ))
            if isfile(analysis_path) && mtime(analysis_path) >= mtime(path)
                println("Skipping completed $(spec.id) ($architecture)")
                continue
            end
            deleteat!(results, row_index)
        end

        println("Analysing $(spec.id) ($architecture)...")
        model = build_model(spec)
        model_state = JLD2.load(path, "model_state")
        Flux.loadmodel!(model, model_state)
        tropical_network = model_to_tropical(model, spec)

        elapsed = @elapsed regions = linear_regions(
            tropical_network;
            mode = REGION_MODE,
            workers = WORKER_IDS,
        )
        num_regions = length(regions)
        pieces = spec.activation == :maxout ? spec.pieces : missing
        push!(results, (
            spec.id,
            string(spec.activation),
            architecture,
            length(spec.widths),
            first(spec.widths),
            pieces,
            num_regions,
            elapsed,
            "HiGHS",
            "Float64",
        ))
        save_analysis(
            analysis_path;
            num_regions,
            elapsed,
            architecture,
            algorithm = "HiGHS",
            encoding = "Float64",
        )
        write_results(results)
        println("  $num_regions regions in $(round(elapsed; digits = 2)) seconds")
    end
end

run_analysis()
