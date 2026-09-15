module VolumeDynamicsExperiment

import Flux
import Distributed
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

"""Standardized train, validation, and test splits plus training-set statistics."""
struct BinaryDataset
    train::BinarySplit
    validation::BinarySplit
    test::BinarySplit
    feature_mean::Vector{Float64}
    feature_scale::Vector{Float64}
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
    epochs::Int = 200
    seeds::Vector{Int} = [20260827]
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
    generate_dataset(generator, rng; train_size, validation_size, test_size)

Generate independent train, validation, and test splits. Standardize every split
using training-set statistics.
"""
function generate_dataset(
        generator::AbstractBinaryDataGenerator,
        rng::Random.AbstractRNG;
        train_size::Integer,
        validation_size::Integer,
        test_size::Integer,
)
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

    return BinaryDataset(
        train,
        validation,
        test,
        feature_mean,
        feature_scale,
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
    config.epochs > 0 || throw(ArgumentError("epoch count must be positive"))
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
        epochs = _parse_int_env("VOLUME_EPOCHS", 200),
        seeds = _parse_seeds(get(ENV, "VOLUME_SEEDS", "20260827")),
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
        "epochs" => config.epochs,
        "steps_per_epoch" => cld(config.train_size, config.batch_size),
        "checkpoint_cadence" => "every_epoch",
        "analysis_domain" => "whole_plane",
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

function _checkpoint_path(run_path, epoch::Integer)
    return joinpath(run_path, "checkpoints", lpad(string(epoch), 8, '0'))
end

function _save_checkpoint(model, run_path, epoch::Integer, step::Integer, examples_seen::Integer)
    dense_layers = filter(layer -> layer isa Flux.Dense, model.layers)
    weights = [Rational{BigInt}.(layer.weight) for layer in dense_layers]
    biases = [Rational{BigInt}.(layer.bias) for layer in dense_layers]
    checkpoint_path = _checkpoint_path(run_path, epoch)
    mkpath(checkpoint_path)
    temporary_path = joinpath(checkpoint_path, "parameters.jld2.tmp")
    JLD2.jldsave(
        temporary_path;
        weights,
        biases,
        epoch = Int(epoch),
        step = Int(step),
        examples_seen = Int(examples_seen),
    )
    mv(temporary_path, joinpath(checkpoint_path, "parameters.jld2"); force = true)
end

function _empty_training_data()
    return Dict{String, Any}(
        "epoch" => Int[],
        "step" => Int[],
        "examples_seen" => Int[],
        "train_loss" => Float64[],
        "validation_loss" => Float64[],
        "train_accuracy" => Float64[],
        "validation_accuracy" => Float64[],
        "parameter_norm" => Float64[],
    )
end

function _record_checkpoint!(
        training_data,
        model,
        dataset,
        run_path,
        epoch,
        step,
        examples_seen,
)
    train_metrics = _metrics(model, dataset.train)
    validation_metrics = _metrics(model, dataset.validation)
    push!(training_data["epoch"], epoch)
    push!(training_data["step"], step)
    push!(training_data["examples_seen"], examples_seen)
    push!(training_data["train_loss"], train_metrics.loss)
    push!(training_data["validation_loss"], validation_metrics.loss)
    push!(training_data["train_accuracy"], train_metrics.accuracy)
    push!(training_data["validation_accuracy"], validation_metrics.accuracy)
    push!(training_data["parameter_norm"], _parameter_norm(model))
    _save_checkpoint(model, run_path, epoch, step, examples_seen)
    JLD2.jldsave(joinpath(run_path, "training_data.jld2"); training_data)
    @printf(
        "  epoch %d  step %d  loss %.4f  train %.4f  validation %.4f\n",
        epoch,
        step,
        train_metrics.loss,
        train_metrics.accuracy,
        validation_metrics.accuracy,
    )
end

"""
    train_model!(model, dataset, run_path; batch_size, learning_rate,
                 weight_decay, epochs, rng)

Train a Float64 logit model for a fixed number of complete epochs. Save the
untrained model at epoch zero and an exact-rational checkpoint after every
epoch. The test split is evaluated only once, after training.
"""
function train_model!(
        model,
        dataset::BinaryDataset,
        run_path;
        batch_size::Integer,
        learning_rate::Real,
        weight_decay::Real,
        epochs::Integer,
        rng::Random.AbstractRNG,
)
    batch_size > 0 || throw(ArgumentError("batch size must be positive"))
    learning_rate > 0 || throw(ArgumentError("learning rate must be positive"))
    weight_decay >= 0 || throw(ArgumentError("weight decay must be nonnegative"))
    epochs > 0 || throw(ArgumentError("epoch count must be positive"))

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
    _record_checkpoint!(training_data, model, dataset, run_path, 0, step, examples_seen)

    for epoch in 1:epochs
        for (features, labels) in loader
            _, gradients = Flux.withgradient(model) do current_model
                Flux.logitbinarycrossentropy(current_model(features), labels)
            end
            Flux.update!(optimizer_state, model, gradients[1])
            step += 1
            examples_seen += size(labels, 2)
        end
        _record_checkpoint!(
            training_data,
            model,
            dataset,
            run_path,
            epoch,
            step,
            examples_seen,
        )
    end

    test_metrics = _metrics(model, dataset.test)
    final_metrics = Dict{String, Any}(
        "epoch" => epochs,
        "step" => step,
        "test_loss" => test_metrics.loss,
        "test_accuracy" => test_metrics.accuracy,
    )
    JLD2.jldsave(joinpath(run_path, "final_metrics.jld2"); final_metrics)
    @printf("  final test loss %.4f  accuracy %.4f\n", test_metrics.loss, test_metrics.accuracy)
    return training_data, final_metrics
end

"""Save the dataset arrays and normalization without custom types."""
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

function _checkpoint_epochs(run_path)
    checkpoint_root = joinpath(run_path, "checkpoints")
    isdir(checkpoint_root) || return Int[]
    names = filter(readdir(checkpoint_root)) do name
        occursin(r"^\d+$", name) && isdir(joinpath(checkpoint_root, name))
    end
    return sort(parse.(Int, names))
end

"""Return per-region and per-polyhedron combinatorial data."""
function _polyhedral_data(linear_regions)
    data = Dict{String, Vector}(
        "polyhedra_per_region" => Int[],
        "bounded" => Bool[],
        "vertices_per_region" => Int[],
        "vertices_per_polyhedron" => Int[],
        "facets_per_polyhedron" => Int[],
        "rays_per_polyhedron" => Int[],
    )
    for components in values(linear_regions), polyhedra in components
        push!(data["polyhedra_per_region"], length(polyhedra))
        push!(data["bounded"], all(Oscar.is_bounded, polyhedra))
        region_vertices = Set{Tuple}()
        for polyhedron in polyhedra
            vertices = collect(Oscar.vertices(polyhedron))
            union!(region_vertices, Tuple.(vertices))
            push!(data["vertices_per_polyhedron"], length(vertices))
            push!(data["facets_per_polyhedron"], Oscar.n_facets(polyhedron))
            push!(data["rays_per_polyhedron"], Oscar.n_rays(polyhedron))
        end
        push!(data["vertices_per_region"], length(region_vertices))
    end
    return data
end


"""
Analyze every rational checkpoint over the whole input plane. The canonical
`graph.jld2` includes bounded and unbounded regions without domain clipping.
Hoffman constants use the pruned rational signomial and exhaustive matrix
subsets. The library evaluates the Hoffman LPs and rank tests in floating point.
"""
function _analyze_checkpoint(run_path, epoch, mode)
    checkpoint_path = _checkpoint_path(run_path, epoch)
    parameters = JLD2.load(joinpath(checkpoint_path, "parameters.jld2"))
    Int(parameters["epoch"]) == epoch || throw(ArgumentError(
        "checkpoint directory $epoch contains parameters for epoch " *
        string(parameters["epoch"]),
    ))
    step = Int(parameters["step"])
    weights = parameters["weights"]
    biases = parameters["biases"]
    thresholds = [zeros(Rational{BigInt}, length(bias)) for bias in biases[1:(end - 1)]]

    pre_pruning = tropicalize(weights, biases, thresholds)[1]
    # Each epoch is analyzed by one process. Do not nest another worker pool here.
    post_pruning = TropicalNN.prune(pre_pruning; mode = mode)
    regions = TropicalNN.map_statistic(identity, post_pruning; mode = mode)
    graph = TropicalNN.get_graph(regions)
    polyhedral_data = _polyhedral_data(regions)
    edge_data = Dict(
        "directions" => TropicalNN._edge_directions(graph)["full"],
        "lengths" => TropicalNN._edge_lengths(graph)["full"],
    )

    JLD2.jldsave(joinpath(checkpoint_path, "graph.jld2"); graph)
    JLD2.jldsave(joinpath(checkpoint_path, "edge_data.jld2"); edge_data)
    JLD2.jldsave(joinpath(checkpoint_path, "polyhedral_data.jld2"); polyhedral_data)
    hoffman = hoffman_constant(post_pruning; brute_force = true, mode = mode)
    GC.gc()
    return (
        epoch = epoch,
        step = step,
        pre = monomial_count(pre_pruning),
        post = monomial_count(post_pruning),
        hoffman_constant = Float64(hoffman),
    )
end

"""Analyze checkpoints sequentially per epoch and in parallel across epochs when workers exist."""
function analyze_checkpoints(run_path; mode, workers = nothing)
    epochs = _checkpoint_epochs(run_path)
    isempty(epochs) && throw(ArgumentError("no checkpoints found in $run_path"))

    analyze_epoch = epoch -> _analyze_checkpoint(run_path, epoch, mode)
    results = workers === nothing ? map(analyze_epoch, epochs) :
        Distributed.pmap(analyze_epoch, workers, epochs)
    sort!(results; by = result -> result.epoch)

    monomial_data = Dict{String, Any}(
        "epoch" => [result.epoch for result in results],
        "step" => [result.step for result in results],
        "pre" => [result.pre for result in results],
        "post" => [result.post for result in results],
    )
    checkpoint_data = Dict{String, Any}(
        "epoch" => [result.epoch for result in results],
        "step" => [result.step for result in results],
        "hoffman_constant" => [result.hoffman_constant for result in results],
        "hoffman_algorithm" => "brute_force",
        "hoffman_norm" => "infinity",
        "hoffman_representation" => "pruned_rational_signomial",
        "hoffman_evaluation" => "floating_point_lp_and_rank_tests",
    )
    JLD2.jldsave(joinpath(run_path, "monomial_data.jld2"); monomial_data)
    JLD2.jldsave(joinpath(run_path, "checkpoint_data.jld2"); checkpoint_data)
    return monomial_data
end
_expected_checkpoint_epochs(config::ExperimentConfig) = collect(0:config.epochs)

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
        "checkpoint_data.jld2",
    )
    all(name -> isfile(joinpath(run_path, name)), required_files) || return false
    saved_config = JLD2.load(joinpath(run_path, "config.jld2"))["config_data"]
    saved_config == _config_data(config, seed) || return false
    epochs = _checkpoint_epochs(run_path)
    epochs == _expected_checkpoint_epochs(config) || return false
    return all(epochs) do epoch
        checkpoint_path = _checkpoint_path(run_path, epoch)
        all(name -> isfile(joinpath(checkpoint_path, name)), (
            "parameters.jld2",
            "graph.jld2",
            "edge_data.jld2",
            "polyhedral_data.jld2",
        ))
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
        epochs = config.epochs,
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
