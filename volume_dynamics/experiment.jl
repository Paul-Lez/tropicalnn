module VolumeDynamicsExperiment

import Flux
using JLD2
import MLUtils
import Oscar
import Plots
using Printf
import Random
using Statistics
using TropicalNN

export AbstractBinaryDataGenerator,
       BinaryDataset,
       BinarySplit,
       ExperimentConfig,
       LinearBandsGenerator,
       SpiralGenerator,
       analyze_checkpoints,
       build_model,
       config_from_env,
       dataset_name,
       generate_dataset,
       generate_split,
       generator_config,
       plot_data_distribution,
       run_experiment,
       save_dataset,
       train_model!

"""Interface for reproducible two-dimensional binary-data generators."""
abstract type AbstractBinaryDataGenerator end

"""Two interleaved spiral arms generated with `MLUtils.Datasets.make_spiral`."""
Base.@kwdef struct SpiralGenerator <: AbstractBinaryDataGenerator
    noise::Float64 = 0.04
    turns::Float64 = 1.0
    radius::Float64 = 1.0
end

"""The original linearly separable horizontal-band dataset, centred at the origin."""
Base.@kwdef struct LinearBandsGenerator <: AbstractBinaryDataGenerator
    gap::Float64 = 0.0
end

"""Return the stable output-directory name for a data generator."""
dataset_name(::SpiralGenerator) = "spiral"
dataset_name(::LinearBandsGenerator) = "linear_bands"

"""Return serializable generator settings to store with a run."""
generator_config(::AbstractBinaryDataGenerator) = Dict{String, Any}()
generator_config(generator::SpiralGenerator) = Dict{String, Any}(
    "noise" => generator.noise,
    "turns" => generator.turns,
    "radius" => generator.radius,
)
generator_config(generator::LinearBandsGenerator) = Dict{String, Any}(
    "gap" => generator.gap,
)

"""One binary-data split with observations in rows and labels encoded as 0.0 or 1.0."""
struct BinarySplit
    features::Matrix{Float64}
    labels::Vector{Float64}

    function BinarySplit(features::AbstractMatrix, labels::AbstractVector)
        size(features, 1) == length(labels) || throw(DimensionMismatch(
            "the number of feature rows must match the number of labels"
        ))
        size(features, 2) == 2 || throw(DimensionMismatch(
            "volume dynamics requires two-dimensional features"
        ))
        all(label -> label == 0 || label == 1, labels) || throw(ArgumentError(
            "binary labels must be zero or one"
        ))
        return new(Matrix{Float64}(features), Vector{Float64}(labels))
    end
end

"""Standardized train, validation, and test splits plus their analysis domain."""
struct BinaryDataset
    train::BinarySplit
    validation::BinarySplit
    test::BinarySplit
    feature_mean::Vector{Float64}
    feature_scale::Vector{Float64}
    analysis_lower::Vector{Float64}
    analysis_upper::Vector{Float64}
end

"""Configuration for training and exact analysis of volume dynamics."""
Base.@kwdef struct ExperimentConfig
    generator::AbstractBinaryDataGenerator = SpiralGenerator()
    train_size::Int = 1000
    validation_size::Int = 250
    test_size::Int = 500
    width::Int = 10
    hidden_bias_scale::Float64 = 0.25
    batch_size::Int = 16
    learning_rate::Float64 = 1e-3
    weight_decay::Float64 = 1e-4
    max_steps::Int = 4000
    checkpoint_every::Int = 100
    analysis_margin::Float64 = 0.10
    seeds::Vector{Int} = collect(20260827:20260829)
end

function _validate_sample_count(n_samples::Integer)
    n_samples >= 4 || throw(ArgumentError("a split must contain at least four samples"))
    iseven(n_samples) || throw(ArgumentError(
        "balanced binary splits require an even sample count, got $n_samples"
    ))
    return Int(n_samples)
end

function _shuffle_split(rng::Random.AbstractRNG, features, labels)
    permutation = Random.randperm(rng, length(labels))
    return BinarySplit(features[permutation, :], labels[permutation])
end

"""Generate one balanced, shuffled `BinarySplit` from a data generator."""
function generate_split(
        generator::SpiralGenerator,
        rng::Random.AbstractRNG,
        n_samples::Integer,
)
    n_samples = _validate_sample_count(n_samples)
    generator.noise >= 0 || throw(ArgumentError("spiral noise must be nonnegative"))
    generator.turns > 0 || throw(ArgumentError("spiral turns must be positive"))
    generator.radius > 0 || throw(ArgumentError("spiral radius must be positive"))

    samples_per_class = n_samples ÷ 2
    theta = (samples_per_class - 1) / (2 * generator.turns)
    coordinates, labels = MLUtils.Datasets.make_spiral(
        samples_per_class,
        generator.radius,
        theta,
        Float64(samples_per_class);
        noise = 0.0,
    )
    coordinates .+= generator.noise .* Random.randn(rng, size(coordinates))
    return _shuffle_split(rng, permutedims(coordinates), labels)
end

function generate_split(
        generator::LinearBandsGenerator,
        rng::Random.AbstractRNG,
        n_samples::Integer,
)
    n_samples = _validate_sample_count(n_samples)
    0 <= generator.gap < 1 || throw(ArgumentError("the band gap must lie in [0, 1)"))

    samples_per_class = n_samples ÷ 2
    features = Matrix{Float64}(undef, n_samples, 2)
    labels = vcat(zeros(Int, samples_per_class), ones(Int, samples_per_class))
    features[:, 1] .= 2 .* Random.rand(rng, n_samples) .- 1
    vertical_span = 1 - generator.gap
    features[1:samples_per_class, 2] .=
        generator.gap .+ vertical_span .* Random.rand(rng, samples_per_class)
    features[(samples_per_class + 1):end, 2] .=
        .-generator.gap .- vertical_span .* Random.rand(rng, samples_per_class)
    return _shuffle_split(rng, features, labels)
end

function _standardize(split::BinarySplit, feature_mean, feature_scale)
    features = (split.features .- permutedims(feature_mean)) ./ permutedims(feature_scale)
    return BinarySplit(features, split.labels)
end

"""
    generate_dataset(generator, rng; train_size, validation_size, test_size,
                     analysis_margin=0.1)

Generate independent train, validation, and test splits. Standardize every split
using training-set statistics and return a bounding box containing all observations.
"""
function generate_dataset(
        generator::AbstractBinaryDataGenerator,
        rng::Random.AbstractRNG;
        train_size::Integer,
        validation_size::Integer,
        test_size::Integer,
        analysis_margin::Real = 0.10,
)
    analysis_margin >= 0 || throw(ArgumentError("analysis margin must be nonnegative"))
    raw_train = generate_split(generator, rng, train_size)
    raw_validation = generate_split(generator, rng, validation_size)
    raw_test = generate_split(generator, rng, test_size)

    feature_mean = vec(Statistics.mean(raw_train.features; dims = 1))
    feature_scale = vec(Statistics.std(raw_train.features; dims = 1, corrected = false))
    all(>(0), feature_scale) || throw(ArgumentError(
        "training features must have positive variance"
    ))

    train = _standardize(raw_train, feature_mean, feature_scale)
    validation = _standardize(raw_validation, feature_mean, feature_scale)
    test = _standardize(raw_test, feature_mean, feature_scale)

    all_features = vcat(train.features, validation.features, test.features)
    lower = vec(minimum(all_features; dims = 1))
    upper = vec(maximum(all_features; dims = 1))
    padding = Float64(analysis_margin) .* (upper .- lower)
    return BinaryDataset(
        train,
        validation,
        test,
        feature_mean,
        feature_scale,
        lower .- padding,
        upper .+ padding,
    )
end

function _validate_config(config::ExperimentConfig)
    _validate_sample_count(config.train_size)
    _validate_sample_count(config.validation_size)
    _validate_sample_count(config.test_size)
    config.width > 0 || throw(ArgumentError("network width must be positive"))
    config.hidden_bias_scale >= 0 || throw(ArgumentError("hidden bias scale must be nonnegative"))
    config.batch_size > 0 || throw(ArgumentError("batch size must be positive"))
    config.learning_rate > 0 || throw(ArgumentError("learning rate must be positive"))
    config.weight_decay >= 0 || throw(ArgumentError("weight decay must be nonnegative"))
    config.max_steps > 0 || throw(ArgumentError("maximum step count must be positive"))
    config.checkpoint_every > 0 || throw(ArgumentError("checkpoint interval must be positive"))
    config.analysis_margin >= 0 || throw(ArgumentError("analysis margin must be nonnegative"))
    isempty(config.seeds) && throw(ArgumentError("at least one seed is required"))
    all(>=(0), config.seeds) || throw(ArgumentError("seeds must be nonnegative"))
    allunique(config.seeds) || throw(ArgumentError("seeds must be unique"))
    return config
end

function _parse_int_env(name, default)
    return parse(Int, get(ENV, name, string(default)))
end

function _parse_float_env(name, default)
    return parse(Float64, get(ENV, name, string(default)))
end

function _parse_seeds(value)
    seeds = parse.(Int, strip.(split(value, ',')))
    isempty(seeds) && throw(ArgumentError("VOLUME_SEEDS must contain at least one seed"))
    return seeds
end

function _generator_from_env()
    name = lowercase(get(ENV, "VOLUME_DATASET", "spiral"))
    if name == "spiral"
        return SpiralGenerator(
            noise = _parse_float_env("VOLUME_SPIRAL_NOISE", 0.04),
            turns = _parse_float_env("VOLUME_SPIRAL_TURNS", 1.0),
            radius = _parse_float_env("VOLUME_SPIRAL_RADIUS", 1.0),
        )
    elseif name in ("linear_bands", "bands")
        return LinearBandsGenerator(gap = _parse_float_env("VOLUME_BAND_GAP", 0.0))
    end
    throw(ArgumentError(
        "unknown VOLUME_DATASET=\"$name\"; expected \"spiral\" or \"linear_bands\""
    ))
end

"""Construct an experiment configuration from `VOLUME_*` environment variables."""
function config_from_env()
    config = ExperimentConfig(
        generator = _generator_from_env(),
        train_size = _parse_int_env("VOLUME_TRAIN_SIZE", 1000),
        validation_size = _parse_int_env("VOLUME_VALIDATION_SIZE", 250),
        test_size = _parse_int_env("VOLUME_TEST_SIZE", 500),
        width = _parse_int_env("VOLUME_WIDTH", 10),
        hidden_bias_scale = _parse_float_env("VOLUME_HIDDEN_BIAS_SCALE", 0.25),
        batch_size = _parse_int_env("VOLUME_BATCH_SIZE", 16),
        learning_rate = _parse_float_env("VOLUME_LEARNING_RATE", 1e-3),
        weight_decay = _parse_float_env("VOLUME_WEIGHT_DECAY", 1e-4),
        max_steps = _parse_int_env("VOLUME_MAX_STEPS", 4000),
        checkpoint_every = _parse_int_env("VOLUME_CHECKPOINT_EVERY", 100),
        analysis_margin = _parse_float_env("VOLUME_ANALYSIS_MARGIN", 0.10),
        seeds = _parse_seeds(get(ENV, "VOLUME_SEEDS", "20260827,20260828,20260829")),
    )
    return _validate_config(config)
end

function _config_data(config::ExperimentConfig, seed::Integer)
    return Dict{String, Any}(
        "dataset" => dataset_name(config.generator),
        "generator" => generator_config(config.generator),
        "train_size" => config.train_size,
        "validation_size" => config.validation_size,
        "test_size" => config.test_size,
        "width" => config.width,
        "hidden_bias_scale" => config.hidden_bias_scale,
        "batch_size" => config.batch_size,
        "learning_rate" => config.learning_rate,
        "weight_decay" => config.weight_decay,
        "max_steps" => config.max_steps,
        "checkpoint_every" => config.checkpoint_every,
        "analysis_margin" => config.analysis_margin,
        "seed" => Int(seed),
        "optimizer" => "AdamW",
        "loss" => "logitbinarycrossentropy",
        "training_number_type" => "Float64",
        "analysis_number_type" => "Rational{BigInt}",
        "julia_version" => string(VERSION),
        "flux_version" => string(Base.pkgversion(Flux)),
        "tropicalnn_version" => string(Base.pkgversion(TropicalNN)),
    )
end

"""Build a two-layer ReLU logit model with Float64 parameters."""
function build_model(
        rng::Random.AbstractRNG,
        width::Integer;
        hidden_bias_scale::Real = 0.25,
)
    width > 0 || throw(ArgumentError("network width must be positive"))
    hidden_bias_scale >= 0 || throw(ArgumentError("hidden bias scale must be nonnegative"))
    hidden_weights = sqrt(2 / 2) .* Random.randn(rng, Float64, width, 2)
    hidden_bias = Float64(hidden_bias_scale) .* Random.randn(rng, Float64, width)
    output_weights = sqrt(1 / width) .* Random.randn(rng, Float64, 1, width)
    output_bias = zeros(Float64, 1)
    return Flux.Chain(
        Flux.Dense(hidden_weights, hidden_bias, Flux.relu),
        Flux.Dense(output_weights, output_bias, identity),
    )
end

function _model_input(split::BinarySplit)
    return Matrix(permutedims(split.features)), reshape(split.labels, 1, :)
end

function _metrics(model, split::BinarySplit)
    features, labels = _model_input(split)
    logits = model(features)
    loss = Float64(Flux.logitbinarycrossentropy(logits, labels))
    accuracy = count((logits .>= 0) .== (labels .>= 0.5)) / length(labels)
    return (loss = loss, accuracy = accuracy)
end

function _parameter_norm(model)
    dense_layers = filter(layer -> layer isa Flux.Dense, model.layers)
    return sqrt(sum(sum(abs2, layer.weight) + sum(abs2, layer.bias) for layer in dense_layers))
end

function _checkpoint_path(run_path, step::Integer)
    return joinpath(run_path, "checkpoints", lpad(string(step), 8, '0'))
end

function _save_checkpoint(model, run_path, step::Integer)
    dense_layers = filter(layer -> layer isa Flux.Dense, model.layers)
    weights = [Rational{BigInt}.(layer.weight) for layer in dense_layers]
    biases = [Rational{BigInt}.(layer.bias) for layer in dense_layers]
    checkpoint_path = _checkpoint_path(run_path, step)
    mkpath(checkpoint_path)
    temporary_path = joinpath(checkpoint_path, "parameters.jld2.tmp")
    JLD2.jldsave(temporary_path; weights, biases, step = Int(step))
    mv(temporary_path, joinpath(checkpoint_path, "parameters.jld2"); force = true)
end

function _empty_training_data()
    return Dict{String, Any}(
        "step" => Int[],
        "examples_seen" => Int[],
        "equivalent_epochs" => Float64[],
        "train_loss" => Float64[],
        "validation_loss" => Float64[],
        "train_accuracy" => Float64[],
        "validation_accuracy" => Float64[],
        "parameter_norm" => Float64[],
    )
end

function _record_checkpoint!(training_data, model, dataset, run_path, step, examples_seen)
    train_metrics = _metrics(model, dataset.train)
    validation_metrics = _metrics(model, dataset.validation)
    push!(training_data["step"], step)
    push!(training_data["examples_seen"], examples_seen)
    push!(training_data["equivalent_epochs"], examples_seen / length(dataset.train.labels))
    push!(training_data["train_loss"], train_metrics.loss)
    push!(training_data["validation_loss"], validation_metrics.loss)
    push!(training_data["train_accuracy"], train_metrics.accuracy)
    push!(training_data["validation_accuracy"], validation_metrics.accuracy)
    push!(training_data["parameter_norm"], _parameter_norm(model))
    _save_checkpoint(model, run_path, step)
    JLD2.jldsave(joinpath(run_path, "training_data.jld2"); training_data)
    @printf(
        "  step %d  epochs %.2f  loss %.4f  train %.4f  validation %.4f\n",
        step,
        last(training_data["equivalent_epochs"]),
        train_metrics.loss,
        train_metrics.accuracy,
        validation_metrics.accuracy,
    )
end

"""
    train_model!(model, dataset, run_path; batch_size, learning_rate,
                 weight_decay, max_steps, checkpoint_every, rng)

Train a Float64 logit model for a fixed number of optimizer updates. Save the
untrained model at step zero and exact-rational checkpoints thereafter. The test
split is evaluated only once, after training.
"""
function train_model!(
        model,
        dataset::BinaryDataset,
        run_path;
        batch_size::Integer,
        learning_rate::Real,
        weight_decay::Real,
        max_steps::Integer,
        checkpoint_every::Integer,
        rng::Random.AbstractRNG,
)
    batch_size > 0 || throw(ArgumentError("batch size must be positive"))
    learning_rate > 0 || throw(ArgumentError("learning rate must be positive"))
    weight_decay >= 0 || throw(ArgumentError("weight decay must be nonnegative"))
    max_steps > 0 || throw(ArgumentError("maximum step count must be positive"))
    checkpoint_every > 0 || throw(ArgumentError("checkpoint interval must be positive"))

    train_features, train_labels = _model_input(dataset.train)
    loader = Flux.DataLoader(
        (train_features, train_labels);
        batchsize = batch_size,
        shuffle = true,
        rng = rng,
    )
    optimizer_state = Flux.setup(
        Flux.AdamW(; eta = learning_rate, lambda = weight_decay),
        model,
    )
    training_data = _empty_training_data()
    step = 0
    examples_seen = 0
    _record_checkpoint!(training_data, model, dataset, run_path, step, examples_seen)

    while step < max_steps
        for (features, labels) in loader
            _, gradients = Flux.withgradient(model) do current_model
                Flux.logitbinarycrossentropy(current_model(features), labels)
            end
            Flux.update!(optimizer_state, model, gradients[1])
            step += 1
            examples_seen += size(labels, 2)

            if step % checkpoint_every == 0 || step == max_steps
                _record_checkpoint!(training_data, model, dataset, run_path, step, examples_seen)
            end
            step == max_steps && break
        end
    end

    test_metrics = _metrics(model, dataset.test)
    final_metrics = Dict{String, Any}(
        "step" => step,
        "test_loss" => test_metrics.loss,
        "test_accuracy" => test_metrics.accuracy,
    )
    JLD2.jldsave(joinpath(run_path, "final_metrics.jld2"); final_metrics)
    @printf("  final test loss %.4f  accuracy %.4f\n", test_metrics.loss, test_metrics.accuracy)
    return training_data, final_metrics
end

"""Save the dataset arrays, normalization, and analysis bounds without custom types."""
function save_dataset(path, dataset::BinaryDataset)
    JLD2.jldsave(
        path;
        X_train = dataset.train.features,
        Y_train = dataset.train.labels,
        X_validation = dataset.validation.features,
        Y_validation = dataset.validation.labels,
        X_test = dataset.test.features,
        Y_test = dataset.test.labels,
        feature_mean = dataset.feature_mean,
        feature_scale = dataset.feature_scale,
        analysis_lower = dataset.analysis_lower,
        analysis_upper = dataset.analysis_upper,
    )
end

"""Plot the standardized training, validation, and test distributions."""
function plot_data_distribution(path, dataset::BinaryDataset)
    function class_scatter(split, title)
        figure = Plots.scatter(
            split.features[split.labels .== 0, 1],
            split.features[split.labels .== 0, 2];
            label = "Class 0",
            markerstrokewidth = 0,
            markersize = 3,
            alpha = 0.65,
            title = title,
            xlabel = "x₁",
            ylabel = "x₂",
            aspect_ratio = :equal,
        )
        Plots.scatter!(
            figure,
            split.features[split.labels .== 1, 1],
            split.features[split.labels .== 1, 2];
            label = "Class 1",
            markerstrokewidth = 0,
            markersize = 3,
            alpha = 0.65,
        )
        return figure
    end

    figure = Plots.plot(
        class_scatter(dataset.train, "Training data"),
        class_scatter(dataset.validation, "Validation data"),
        class_scatter(dataset.test, "Test data");
        layout = (1, 3),
        size = (1400, 430),
    )
    Plots.savefig(figure, path)
end

function _checkpoint_steps(run_path)
    checkpoint_root = joinpath(run_path, "checkpoints")
    isdir(checkpoint_root) || return Int[]
    names = filter(readdir(checkpoint_root)) do name
        occursin(r"^\d+$", name) && isdir(joinpath(checkpoint_root, name))
    end
    return sort(parse.(Int, names))
end

function _domain_polyhedron(lower, upper)
    length(lower) == length(upper) == 2 || throw(DimensionMismatch(
        "the analysis domain must be two-dimensional"
    ))
    all(lower .< upper) || throw(ArgumentError("analysis lower bounds must be below upper bounds"))
    matrix = Rational{BigInt}[1 0; -1 0; 0 1; 0 -1]
    vector = Rational{BigInt}.([upper[1], -lower[1], upper[2], -lower[2]])
    return Oscar.polyhedron(matrix, vector)
end

function _restrict_to_domain(f, lower, upper; mode)
    regions = TropicalNN.map_statistic(identity, f; mode = mode)
    domain = _domain_polyhedron(lower, upper)
    restricted = Dict{Any, Any}()
    for (linear_map, components) in regions
        restricted_components = Vector{Vector{Any}}()
        for component in components
            restricted_polys = Any[]
            for polyhedron in component
                intersection = Oscar.intersect(polyhedron, domain)
                if Oscar.is_feasible(intersection) && Oscar.is_fulldimensional(intersection)
                    push!(restricted_polys, intersection)
                end
            end
            isempty(restricted_polys) || push!(restricted_components, restricted_polys)
        end
        isempty(restricted_components) || (restricted[linear_map] = restricted_components)
    end
    return restricted
end

"""Analyze every rational checkpoint inside the saved data-domain bounding box."""
function analyze_checkpoints(run_path; mode, workers = nothing)
    dataset_data = JLD2.load(joinpath(run_path, "dataset.jld2"))
    lower = dataset_data["analysis_lower"]
    upper = dataset_data["analysis_upper"]
    steps = _checkpoint_steps(run_path)
    isempty(steps) && throw(ArgumentError("no checkpoints found in $run_path"))

    monomial_data = Dict{String, Any}(
        "step" => Int[],
        "pre" => Int[],
        "post" => Int[],
    )
    for step in steps
        println("  analyzing step $step")
        checkpoint_path = _checkpoint_path(run_path, step)
        parameters = JLD2.load(joinpath(checkpoint_path, "parameters.jld2"))
        weights = parameters["weights"]
        biases = parameters["biases"]
        thresholds = [zeros(Rational{BigInt}, length(bias)) for bias in biases[1:(end - 1)]]

        pre_pruning = tropicalize(weights, biases, thresholds)[1]
        post_pruning = TropicalNN.prune(pre_pruning; mode = mode, workers = workers)
        restricted_regions = _restrict_to_domain(post_pruning, lower, upper; mode = mode)
        graph = TropicalNN.get_graph(restricted_regions)
        edge_data = Dict(
            "directions" => TropicalNN._edge_directions(graph)["full"],
            "lengths" => TropicalNN._edge_lengths(graph)["full"],
        )

        JLD2.jldsave(joinpath(checkpoint_path, "graph.jld2"); graph)
        JLD2.jldsave(joinpath(checkpoint_path, "edge_data.jld2"); edge_data)
        push!(monomial_data["step"], step)
        push!(monomial_data["pre"], monomial_count(pre_pruning))
        push!(monomial_data["post"], monomial_count(post_pruning))
    end
    JLD2.jldsave(joinpath(run_path, "monomial_data.jld2"); monomial_data)
    return monomial_data
end

function _prepare_run_directory(run_path)
    if ispath(run_path)
        isempty(readdir(run_path)) || throw(ArgumentError(
            "$run_path already contains an experiment; choose a new VOLUME_OUTPUT_DIR " *
            "or archive the existing run"
        ))
    end
    mkpath(run_path)
end

function _prepare_output_root(output_root, config::ExperimentConfig)
    mkpath(output_root)
    root_config_path = joinpath(output_root, "experiment_config.jld2")
    root_config = _config_data(config, first(config.seeds))
    delete!(root_config, "seed")
    root_config["seeds"] = copy(config.seeds)
    if isfile(root_config_path)
        saved_config = JLD2.load(root_config_path)["root_config"]
        saved_config == root_config || throw(ArgumentError(
            "$output_root contains runs from a different configuration; choose a new " *
            "VOLUME_OUTPUT_DIR"
        ))
    else
        JLD2.jldsave(root_config_path; root_config)
    end
end

function _completed_run_matches(run_path, config::ExperimentConfig, seed::Integer)
    required_files = (
        "config.jld2",
        "dataset.jld2",
        "training_data.jld2",
        "final_metrics.jld2",
        "monomial_data.jld2",
    )
    all(name -> isfile(joinpath(run_path, name)), required_files) || return false
    saved_config = JLD2.load(joinpath(run_path, "config.jld2"))["config_data"]
    saved_config == _config_data(config, seed) || return false
    steps = _checkpoint_steps(run_path)
    isempty(steps) && return false
    return all(steps) do step
        isfile(joinpath(_checkpoint_path(run_path, step), "graph.jld2"))
    end
end

function _seed_rng(seed::Integer, stream::Integer)
    return Random.MersenneTwister(seed + 1_000_000 * stream)
end

function _run_seed(config, seed, run_path; mode, workers)
    _prepare_run_directory(run_path)
    config_data = _config_data(config, seed)
    JLD2.jldsave(joinpath(run_path, "config.jld2"); config_data)

    dataset = generate_dataset(
        config.generator,
        _seed_rng(seed, 0);
        train_size = config.train_size,
        validation_size = config.validation_size,
        test_size = config.test_size,
        analysis_margin = config.analysis_margin,
    )
    save_dataset(joinpath(run_path, "dataset.jld2"), dataset)
    plot_data_distribution(joinpath(run_path, "data_distribution.png"), dataset)

    model = build_model(
        _seed_rng(seed, 1),
        config.width;
        hidden_bias_scale = config.hidden_bias_scale,
    )
    train_model!(
        model,
        dataset,
        run_path;
        batch_size = config.batch_size,
        learning_rate = config.learning_rate,
        weight_decay = config.weight_decay,
        max_steps = config.max_steps,
        checkpoint_every = config.checkpoint_every,
        rng = _seed_rng(seed, 2),
    )
    analyze_checkpoints(run_path; mode = mode, workers = workers)
    return run_path
end

"""Run all configured seeds and keep each run in an isolated output directory."""
function run_experiment(
        config::ExperimentConfig;
        output_root = joinpath("outputs", "volume_dynamics", dataset_name(config.generator)),
        mode,
    workers = nothing,
)
    _validate_config(config)
    _prepare_output_root(output_root, config)
    run_paths = String[]
    for seed in config.seeds
        run_path = joinpath(output_root, "seed_$seed")
        if isdir(run_path) && _completed_run_matches(run_path, config, seed)
            println("Skipping completed $(dataset_name(config.generator)) run with seed $seed")
            push!(run_paths, run_path)
            continue
        end
        println("Running $(dataset_name(config.generator)) volume dynamics with seed $seed")
        push!(run_paths, _run_seed(config, seed, run_path; mode = mode, workers = workers))
    end
    return run_paths
end

end
