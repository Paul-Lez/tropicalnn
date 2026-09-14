using CSV
using DataFrames
using Flux
using Flux: DataLoader, crossentropy, onecold, onehotbatch
using JLD2
using MLDatasets
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "experiment_setup.jl"))
include(joinpath(@__DIR__, "models.jl"))
using .MNISTModels

const OUTPUT_DIR = joinpath("outputs", "mnist")
const METRICS_PATH = joinpath(OUTPUT_DIR, "metrics.csv")
const METRICS_SUMMARY_PATH = joinpath(OUTPUT_DIR, "metrics_summary.csv")

function get_data(batch_size)
    x_train, y_train = MLDatasets.MNIST.traindata(Float32)
    x_test, y_test = MLDatasets.MNIST.testdata(Float32)

    x_train = reshape(Float32.(x_train), 28^2, :)
    y_train = onehotbatch(y_train, 0:9)
    x_test = reshape(Float32.(x_test), 28^2, :)
    y_test = onehotbatch(y_test, 0:9)

    train_loader = DataLoader((x_train, y_train); batchsize = batch_size, shuffle = true)
    return train_loader, (x_train, y_train), (x_test, y_test)
end

accuracy(model, data) = mean(onecold(model(data[1])) .== onecold(data[2]))

function train_model!(model, train_loader, epochs, learning_rate)
    optimizer_state = Flux.setup(Flux.Adam(learning_rate), model)
    for epoch in 1:epochs
        loss_sum = 0.0
        batch_count = 0
        for (input, target) in train_loader
            loss_value, gradients = Flux.withgradient(model) do current_model
                crossentropy(current_model(input), target)
            end
            Flux.update!(optimizer_state, model, gradients[1])
            loss_sum += loss_value
            batch_count += 1
        end
        @printf("  epoch %d/%d, average loss %.4f\n", epoch, epochs, loss_sum / batch_count)
    end
    return model
end

function empty_metrics()
    return DataFrame(
        Model = String[],
        Sample = Int[],
        Activation = String[],
        Architecture = String[],
        HiddenLayers = Int[],
        Width = Int[],
        Pieces = Union{Missing, Int}[],
        Epochs = Int[],
        BatchSize = Int[],
        LearningRate = Float64[],
        Seed = Int[],
        TrainAccuracy = Float64[],
        TestAccuracy = Float64[],
    )
end

function load_metrics()
    isfile(METRICS_PATH) || return empty_metrics()
    return load_typed_csv(METRICS_PATH, empty_metrics())
end

function metrics_summary(metrics, num_samples)
    summary = combine(
        groupby(metrics, [
            :Model,
            :Activation,
            :Architecture,
            :HiddenLayers,
            :Width,
            :Pieces,
            :Epochs,
            :BatchSize,
            :LearningRate,
        ]),
        :Sample => length => :NumSamples,
        :TrainAccuracy => mean => :MeanTrainAccuracy,
        :TrainAccuracy => std => :StdTrainAccuracy,
        :TestAccuracy => mean => :MeanTestAccuracy,
        :TestAccuracy => std => :StdTestAccuracy,
    )
    filter!(:NumSamples => ==(num_samples), summary)
    return summary
end

function write_metrics(metrics, num_samples)
    temporary_path = METRICS_PATH * ".tmp"
    CSV.write(temporary_path, metrics)
    mv(temporary_path, METRICS_PATH; force = true)

    temporary_summary_path = METRICS_SUMMARY_PATH * ".tmp"
    CSV.write(temporary_summary_path, metrics_summary(metrics, num_samples))
    mv(temporary_summary_path, METRICS_SUMMARY_PATH; force = true)
end

function save_model_state(path, model_state)
    temporary_path = path * ".tmp"
    jldsave(temporary_path; model_state)
    mv(temporary_path, path; force = true)
end

function run_experiment()
    epochs = parse(Int, get(ENV, "MNIST_EPOCHS", "100"))
    batch_size = parse(Int, get(ENV, "MNIST_BATCH_SIZE", "128"))
    learning_rate = parse(Float64, get(ENV, "MNIST_LEARNING_RATE", "0.005"))
    num_samples = parse(Int, get(
        ENV,
        "MNIST_SAMPLES",
        string(DEFAULT_NUM_SAMPLES),
    ))
    epochs > 0 || throw(ArgumentError("MNIST_EPOCHS must be positive"))
    batch_size > 0 || throw(ArgumentError("MNIST_BATCH_SIZE must be positive"))
    num_samples > 0 || throw(ArgumentError("MNIST_SAMPLES must be positive"))

    mkpath(joinpath(OUTPUT_DIR, "models"))
    metrics = load_metrics()
    specs = experiment_specs()
    pending_specs = Tuple{Int, Int, ExperimentSpec}[]

    for (spec_index, spec) in enumerate(specs)
        for sample in 1:num_samples
            architecture = join(vcat(28^2, spec.widths, 10), ":")
            seed = 20260824 + spec_index + (sample - 1) * length(specs)
            matching_rows = findall(
                (metrics.Model .== spec.id) .& (metrics.Sample .== sample)
            )
            length(matching_rows) <= 1 || throw(ArgumentError(
                "$METRICS_PATH contains duplicate rows for $(spec.id), sample $sample"
            ))
            if !isempty(matching_rows)
                row_index = only(matching_rows)
                row = metrics[row_index, :]
                expected_pieces = spec.activation == :maxout ? spec.pieces : missing
                configuration_matches = row.Activation == string(spec.activation) &&
                    row.Architecture == architecture &&
                    row.HiddenLayers == length(spec.widths) &&
                    row.Width == first(spec.widths) &&
                    isequal(row.Pieces, expected_pieces) &&
                    row.Epochs == epochs &&
                    row.BatchSize == batch_size &&
                    row.LearningRate == learning_rate &&
                    row.Seed == seed
                configuration_matches || throw(ArgumentError(
                    "saved configuration for $(spec.id), sample $sample does not match this run; " *
                    "move or remove $METRICS_PATH and outputs/mnist/models before rerunning"
                ))
                if isfile(model_path(OUTPUT_DIR, spec, sample))
                    println("Skipping completed $(spec.id), sample $sample ($architecture)")
                    continue
                end
                deleteat!(metrics, row_index)
            end
            push!(pending_specs, (spec_index, sample, spec))
        end
    end

    if isempty(pending_specs)
        write_metrics(metrics, num_samples)
        return
    end
    println("Loading MNIST data...")
    train_loader, train_data, test_data = get_data(batch_size)

    for (spec_index, sample, spec) in pending_specs
        architecture = join(vcat(28^2, spec.widths, 10), ":")
        seed = 20260824 + spec_index + (sample - 1) * length(specs)
        Random.seed!(seed)
        println("Training $(spec.id), sample $sample/$num_samples ($architecture)...")
        model = build_model(spec)
        train_model!(model, train_loader, epochs, learning_rate)
        train_accuracy = accuracy(model, train_data)
        test_accuracy = accuracy(model, test_data)

        model_state = Flux.state(model)
        save_model_state(model_path(OUTPUT_DIR, spec, sample), model_state)
        pieces = spec.activation == :maxout ? spec.pieces : missing
        push!(metrics, (
            spec.id,
            sample,
            string(spec.activation),
            architecture,
            length(spec.widths),
            first(spec.widths),
            pieces,
            epochs,
            batch_size,
            learning_rate,
            seed,
            train_accuracy,
            test_accuracy,
        ))
        write_metrics(metrics, num_samples)
        @printf("  train accuracy %.2f%%, test accuracy %.2f%%\n",
            100 * train_accuracy, 100 * test_accuracy)
    end
end

run_experiment()
