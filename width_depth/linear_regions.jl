include(joinpath(@__DIR__, "..", "experiment_setup.jl"))
const EXPERIMENT_RUNTIME = setup_experiment!()

using CSV
using DataFrames
using Logging
using Random
using Statistics
using TropicalNN

const REGION_MODE = highs_mode(EXPERIMENT_RUNTIME)
const WORKER_IDS = tropical_workers(EXPERIMENT_RUNTIME)
const OUTPUT_PATH = joinpath("outputs", "width_depth", "linear_regions.csv")
const SUMMARY_OUTPUT_PATH = joinpath(
    "outputs",
    "width_depth",
    "linear_regions_summary.csv",
)
const MAXOUT_PIECES = 3

global_logger(SimpleLogger(stderr, Logging.Error))

function relu_network(dims)
    weights, biases, _ = random_mlp(dims; symbolic = false)
    layers = AbstractNeuralNetworkLayer{Float64}[]
    for layer_index in eachindex(weights)
        push!(layers, AffineLayer(weights[layer_index], biases[layer_index]))
        if layer_index < length(weights)
            push!(layers, ActivationLayer(relu(Float64), dims[layer_index + 1]))
        end
    end
    return NeuralNetwork(layers)
end

function warm_up_linear_regions()
    network = relu_network([2, 2, 1])
    linear_regions(network; mode = REGION_MODE, workers = WORKER_IDS)
    return nothing
end

function experiment_architectures()
    architectures = NamedTuple[]
    for hidden_layers in 1:6
        dims = vcat(2, fill(10, hidden_layers), 1)
        push!(architectures, (; sweep = "depth", hidden_layers, width = 10, dims))
    end
    for width in 10:10:60
        dims = [2, width, 1]
        push!(architectures, (; sweep = "width", hidden_layers = 1, width, dims))
    end
    return architectures
end

function empty_results()
    return DataFrame(
        Network = String[],
        Sweep = String[],
        Architecture = String[],
        HiddenLayers = Int[],
        Width = Int[],
        Pieces = Union{Missing, Int}[],
        Trial = Int[],
        Seed = Int[],
        NumRegions = Int[],
        TimeSeconds = Float64[],
        Algorithm = String[],
        Encoding = String[],
    )
end

function load_results()
    isfile(OUTPUT_PATH) || return empty_results()
    results = load_typed_csv(OUTPUT_PATH, empty_results())
    all(results.Algorithm .== "HiGHS") || throw(ArgumentError(
        "$OUTPUT_PATH contains results from an algorithm other than HiGHS"
    ))
    all(results.Encoding .== "Float64") || throw(ArgumentError(
        "$OUTPUT_PATH contains results with an encoding other than Float64"
    ))
    return results
end

function results_summary(results, num_trials)
    summary = combine(
        groupby(results, [
            :Network,
            :Sweep,
            :Architecture,
            :HiddenLayers,
            :Width,
            :Pieces,
            :Algorithm,
            :Encoding,
        ]),
        :Trial => length => :NumSamples,
        :NumRegions => mean => :MeanNumRegions,
        :NumRegions => std => :StdNumRegions,
        :TimeSeconds => mean => :MeanTimeSeconds,
        :TimeSeconds => std => :StdTimeSeconds,
        :TimeSeconds => median => :MedianTimeSeconds,
        :TimeSeconds => minimum => :MinTimeSeconds,
        :TimeSeconds => maximum => :MaxTimeSeconds,
    )
    filter!(:NumSamples => ==(num_trials), summary)
    return summary
end

function write_results(results, num_trials)
    temporary_path = OUTPUT_PATH * ".tmp"
    CSV.write(temporary_path, results)
    mv(temporary_path, OUTPUT_PATH; force = true)

    temporary_summary_path = SUMMARY_OUTPUT_PATH * ".tmp"
    CSV.write(temporary_summary_path, results_summary(results, num_trials))
    mv(temporary_summary_path, SUMMARY_OUTPUT_PATH; force = true)
end

function run_experiment()
    num_trials = parse(Int, get(ENV, "WIDTH_DEPTH_TRIALS", "30"))
    num_trials > 0 || throw(ArgumentError("WIDTH_DEPTH_TRIALS must be positive"))

    mkpath(dirname(OUTPUT_PATH))
    results = load_results()
    completed = Dict(
        (row.Network, row.Sweep, row.Architecture, row.Trial) => row
        for row in eachrow(results)
    )
    length(completed) == nrow(results) || throw(ArgumentError(
        "$OUTPUT_PATH contains duplicate experiment rows"
    ))

    println("Warming up linear-region evaluation...")
    warm_up_linear_regions()

    for (architecture_index, architecture) in enumerate(experiment_architectures())
        dims = architecture.dims
        architecture_name = join(dims, ":")
        for (network_index, network_name) in enumerate(("relu", "maxout"))
            for trial in 1:num_trials
                result_key = (network_name, architecture.sweep, architecture_name, trial)
                seed = 20260824 + 1_000 * architecture_index + 100 * network_index + trial
                if haskey(completed, result_key)
                    row = completed[result_key]
                    expected_pieces = network_name == "maxout" ? MAXOUT_PIECES : missing
                    metadata_matches = row.HiddenLayers == architecture.hidden_layers &&
                        row.Width == architecture.width &&
                        isequal(row.Pieces, expected_pieces) &&
                        row.Seed == seed &&
                        row.Algorithm == "HiGHS" &&
                        row.Encoding == "Float64"
                    metadata_matches || throw(ArgumentError(
                        "$OUTPUT_PATH contains incompatible metadata for $result_key"
                    ))
                    println("Skipping completed $network_name $(architecture.sweep) $architecture_name, trial $trial/$num_trials")
                    continue
                end

                Random.seed!(seed)
                network = network_name == "relu" ?
                    relu_network(dims) :
                    random_maxout_network(dims, MAXOUT_PIECES, Float64)
                @assert network isa NeuralNetwork{Float64}

                println("$network_name $(architecture.sweep) $architecture_name, trial $trial/$num_trials")
                elapsed = @elapsed regions = linear_regions(
                    network;
                    mode = REGION_MODE,
                    workers = WORKER_IDS,
                )
                pieces = network_name == "maxout" ? MAXOUT_PIECES : missing
                push!(results, (
                    network_name,
                    architecture.sweep,
                    architecture_name,
                    architecture.hidden_layers,
                    architecture.width,
                    pieces,
                    trial,
                    seed,
                    length(regions),
                    elapsed,
                    "HiGHS",
                    "Float64",
                ))
                completed[result_key] = last(eachrow(results))
                write_results(results, num_trials)
                println("  $(length(regions)) regions in $(round(elapsed; digits = 2)) seconds")
            end
        end
    end
    write_results(results, num_trials)
end

run_experiment()
