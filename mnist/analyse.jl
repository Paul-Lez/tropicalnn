include(joinpath(@__DIR__, "..", "experiment_setup.jl"))
const EXPERIMENT_RUNTIME = setup_experiment!()

using CSV
using DataFrames
import Flux
using JLD2
using Statistics
using TropicalNN

include(joinpath(@__DIR__, "models.jl"))
using .MNISTModels

const REGION_MODE = highs_mode(EXPERIMENT_RUNTIME)
const WORKER_IDS = tropical_workers(EXPERIMENT_RUNTIME)
const OUTPUT_DIR = joinpath("outputs", "mnist")
const RESULTS_PATH = joinpath(OUTPUT_DIR, "linear_regions.csv")
const RESULTS_SUMMARY_PATH = joinpath(OUTPUT_DIR, "linear_regions_summary.csv")

function empty_results()
    return DataFrame(
        Model = String[],
        Sample = Int[],
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

function results_summary(results, num_samples)
    summary = combine(
        groupby(results, [
            :Model,
            :Activation,
            :Architecture,
            :HiddenLayers,
            :Width,
            :Pieces,
            :Algorithm,
            :Encoding,
        ]),
        :Sample => length => :NumSamples,
        :NumRegions => mean => :MeanNumRegions,
        :NumRegions => std => :StdNumRegions,
        :TimeSeconds => mean => :MeanTimeSeconds,
        :TimeSeconds => std => :StdTimeSeconds,
        :TimeSeconds => median => :MedianTimeSeconds,
        :TimeSeconds => minimum => :MinTimeSeconds,
        :TimeSeconds => maximum => :MaxTimeSeconds,
    )
    filter!(:NumSamples => ==(num_samples), summary)
    return summary
end

function write_results(results, num_samples)
    temporary_path = RESULTS_PATH * ".tmp"
    CSV.write(temporary_path, results)
    mv(temporary_path, RESULTS_PATH; force = true)

    temporary_summary_path = RESULTS_SUMMARY_PATH * ".tmp"
    CSV.write(temporary_summary_path, results_summary(results, num_samples))
    mv(temporary_summary_path, RESULTS_SUMMARY_PATH; force = true)
end

function save_analysis(path; kwargs...)
    temporary_path = path * ".tmp"
    jldsave(temporary_path; kwargs...)
    mv(temporary_path, path; force = true)
end

function warm_up_analysis()
    spec = ExperimentSpec("warmup", :relu, [1])
    model = build_model(spec; input_dimension = 2, output_dimension = 2)
    tropical_network = model_to_tropical(model, spec)
    linear_regions(tropical_network; mode = REGION_MODE, workers = WORKER_IDS)
    return nothing
end

function run_analysis()
    results = load_results()
    analysis_dir = joinpath(OUTPUT_DIR, "analysis")
    mkpath(analysis_dir)
    num_samples = parse(Int, get(
        ENV,
        "MNIST_SAMPLES",
        string(DEFAULT_NUM_SAMPLES),
    ))
    num_samples > 0 || throw(ArgumentError("MNIST_SAMPLES must be positive"))

    println("Warming up linear-region analysis...")
    warm_up_analysis()

    for spec in experiment_specs()
        for sample in 1:num_samples
            architecture = join(vcat(28^2, spec.widths, 10), ":")
            path = model_path(OUTPUT_DIR, spec, sample)
            isfile(path) || throw(ArgumentError(
                "missing trained model: $path; run mnist/main.jl first"
            ))
            analysis_path = joinpath(
                analysis_dir,
                basename(model_path(OUTPUT_DIR, spec, sample)),
            )

            matching_rows = findall(
                (results.Model .== spec.id) .& (results.Sample .== sample)
            )
            length(matching_rows) <= 1 || throw(ArgumentError(
                "$RESULTS_PATH contains duplicate rows for $(spec.id), sample $sample"
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
                    "$RESULTS_PATH contains incompatible metadata for $(spec.id), sample $sample"
                ))
                if isfile(analysis_path) && mtime(analysis_path) >= mtime(path)
                    println("Skipping completed $(spec.id), sample $sample ($architecture)")
                    continue
                end
                deleteat!(results, row_index)
            end

            println("Analysing $(spec.id), sample $sample/$num_samples ($architecture)...")
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
                sample,
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
                sample,
                algorithm = "HiGHS",
                encoding = "Float64",
            )
            write_results(results, num_samples)
            println("  $num_regions regions in $(round(elapsed; digits = 2)) seconds")
        end
    end
    write_results(results, num_samples)
end

run_analysis()
