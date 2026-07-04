using CSV
using DataFrames
using Flux
using Flux: DataLoader, onehotbatch, onecold, crossentropy
using Graphs
using JLD2
using MLDatasets
using Plots
using Printf
using Statistics
using TropicalNN

include("../utils.jl")

const SMOKE_ROOT = joinpath(@__DIR__, "..", "outputs", "smoke")

function run_step(name, f)
    print("smoke: $name ... ")
    flush(stdout)
    result = f()
    println("ok$result")
end

function tiny_binary_data()
    X = Float32[-1.0 1.0; 1.0 1.0; -1.0 -1.0; 1.0 -1.0]
    Y = [0, 1, 0, 1]
    return X, X, Y, Y
end

function model_weights_biases_thresholds(model; symbolic=true)
    num_dense_layers = length(model) - 1
    if symbolic
        weights = [Rational{BigInt}.(model[i].weight) for i in 1:num_dense_layers]
        biases = [Rational{BigInt}.(model[i].bias) for i in 1:num_dense_layers]
        thresholds = [Rational{BigInt}.(zeros(length(model[i].bias))) for i in 1:(num_dense_layers-1)]
    else
        weights = [model[i].weight for i in 1:num_dense_layers]
        biases = [model[i].bias for i in 1:num_dense_layers]
        thresholds = [zeros(length(model[i].bias)) for i in 1:(num_dense_layers-1)]
    end
    return weights, biases, thresholds
end

function smoke_visualize_linear_regions()
    output_dir = joinpath(SMOKE_ROOT, "visualize_linear_regions")
    mkpath(output_dir)

    weights, biases, thresholds = random_mlp([2, 1, 1])
    f = mlp_to_trop(weights, biases, thresholds)[1]
    linear_regions = enum_linear_regions_rat_general(f; mode=OscarMode())

    fig = plot_linear_regions(linear_regions, xlims=(-2.0, 2.0), ylims=(-2.0, 2.0))
    savefig(fig, joinpath(output_dir, "tiny_nn.png"))
    return " ($(length(linear_regions)) regions)"
end

function smoke_effective_radius()
    output_dir = joinpath(SMOKE_ROOT, "effective_radius")
    mkpath(output_dir)

    weights, biases, thresholds = random_mlp([2, 1, 1])
    rmap = mlp_to_trop(weights, biases, thresholds)[1]
    er = exact_er(rmap)
    linear_regions = enum_linear_regions_rat_general(rmap; mode=OscarMode())

    fig = plot_linear_regions(linear_regions, xlims=(-2.0, 2.0), ylims=(-2.0, 2.0))
    savefig(fig, joinpath(output_dir, "bounding_linear_regions.png"))
    return " (er=$(Float64(er)))"
end

function smoke_width_depth()
    output_dir = joinpath(SMOKE_ROOT, "width_depth")
    mkpath(output_dir)

    weights, biases, thresholds = random_mlp([2, 1, 1])
    f_pre = mlp_to_trop(weights, biases, thresholds)[1]
    results = DataFrame(Width=[1], Pre_Avg=[Float64(monomial_count(f_pre))])
    CSV.write(joinpath(output_dir, "sweep_onelayer.csv"), results)
    return " ($(results.Pre_Avg[1]) monomials)"
end

function smoke_get_monomial_counts(model)
    weights, biases, thresholds = model_weights_biases_thresholds(model)
    f_pre = mlp_to_trop(weights, biases, thresholds)[1]
    f_post = monomial_strong_elim(f_pre)
    return monomial_count(f_pre), monomial_count(f_post)
end

function smoke_rate_of_pruning()
    output_dir = joinpath(SMOKE_ROOT, "rate_of_pruning")
    mkpath(output_dir)

    X_train, _, Y_train, _ = tiny_binary_data()
    X_train_mat = Matrix(X_train')
    Y_train_mat = reshape(Y_train, 1, :)

    weights, biases, _ = random_mlp([2, 1, 1])
    model = Chain(Dense(weights[1], biases[1], relu), Dense(weights[2], biases[2], identity), σ)
    pre_init, post_init = smoke_get_monomial_counts(model)

    loader = DataLoader((X_train_mat, Y_train_mat), batchsize=2, shuffle=false)
    loss(m, x, y) = Flux.binarycrossentropy(m(x), y)
    opt_state = Flux.setup(Flux.Adam(1e-3), model)
    for (x_batch, y_batch) in loader
        _, gs = Flux.withgradient(m -> loss(m, x_batch, y_batch), model)
        Flux.update!(opt_state, model, gs[1])
    end

    pre_trained, post_trained = smoke_get_monomial_counts(model)
    results = DataFrame(
        Width = [1],
        Rate_Init = [(pre_init - post_init) / pre_init],
        Rate_Trained = [(pre_trained - post_trained) / pre_trained],
    )
    CSV.write(joinpath(output_dir, "results.csv"), results)
    return " ($(nrow(results)) row)"
end

function smoke_accuracy(model, X, Y)
    X_mat = Matrix(X')
    preds = model(X_mat) .>= 0.5
    return sum(preds .== reshape(Y, 1, :)) / length(Y)
end

function smoke_save_weights(model, path)
    weights = [Rational{BigInt}.(model[i].weight) for i in 1:length(model)-1]
    biases = [Rational{BigInt}.(model[i].bias) for i in 1:length(model)-1]
    mkpath(path)
    JLD2.save(joinpath(path, "weights.jld2"), "data", weights)
    JLD2.save(joinpath(path, "biases.jld2"), "data", biases)
end

function smoke_train_volume_model(data_path, X_train, X_test, Y_train, Y_test)
    weights, biases, _ = random_mlp([2, 1, 1])
    model = Chain(Dense(weights[1], biases[1], relu), Dense(weights[2], biases[2], identity), σ)

    X_train_mat = Matrix(X_train')
    Y_train_mat = reshape(Y_train, 1, :)
    loader = DataLoader((X_train_mat, Y_train_mat), batchsize=2, shuffle=false)
    loss(m, x, y) = Flux.binarycrossentropy(m(x), y)
    opt_state = Flux.setup(Flux.Adam(1e-3), model)

    training_data = Dict("loss"=>Float64[], "train_acc"=>Float64[], "test_acc"=>Float64[])
    for epoch in 0:1
        epoch_loss = 0.0
        count = 0
        for (x_batch, y_batch) in loader
            l, gs = Flux.withgradient(m -> loss(m, x_batch, y_batch), model)
            batch_n = size(y_batch, 2)
            epoch_loss += l * batch_n
            Flux.update!(opt_state, model, gs[1])
            count += batch_n
        end
        push!(training_data["loss"], epoch_loss / count)
        push!(training_data["train_acc"], smoke_accuracy(model, X_train, Y_train))
        push!(training_data["test_acc"], smoke_accuracy(model, X_test, Y_test))
        smoke_save_weights(model, joinpath(data_path, string(epoch)))
    end
    return training_data
end

function smoke_analyze_volume_epochs(data_path)
    epochs = sort(parse.(Int, filter(x -> !occursin(".", x), readdir(data_path))))
    monomial_data = Dict("pre"=>Int[], "post"=>Int[])

    for epoch in epochs
        weights = JLD2.load(joinpath(data_path, string(epoch), "weights.jld2"))["data"]
        biases = JLD2.load(joinpath(data_path, string(epoch), "biases.jld2"))["data"]
        thresholds = [Rational{BigInt}.(zeros(length(bias))) for bias in biases[1:end-1]]

        f_pre = mlp_to_trop(weights, biases, thresholds)[1]
        f_post = monomial_strong_elim(f_pre)
        graph = get_graph(f_post)
        edge_data = Dict("gradients"=>edge_directions(graph)["full"], "lengths"=>edge_lengths(graph)["full"])

        JLD2.save(joinpath(data_path, string(epoch), "graph.jld2"), "graph", graph)
        JLD2.save(joinpath(data_path, string(epoch), "edge_data.jld2"), "data", edge_data)
        push!(monomial_data["pre"], monomial_count(f_pre))
        push!(monomial_data["post"], monomial_count(f_post))
    end

    JLD2.save(joinpath(data_path, "monomial_data.jld2"), "data", monomial_data)
    return monomial_data
end

function smoke_volume_dynamics_main()
    data_path = joinpath(SMOKE_ROOT, "volume_dynamics")
    mkpath(data_path)

    X_train, X_test, Y_train, Y_test = tiny_binary_data()
    JLD2.save(joinpath(data_path, "X_train.jld2"), "data", X_train)
    JLD2.save(joinpath(data_path, "X_test.jld2"), "data", X_test)
    JLD2.save(joinpath(data_path, "Y_train.jld2"), "data", Y_train)
    JLD2.save(joinpath(data_path, "Y_test.jld2"), "data", Y_test)

    training_data = smoke_train_volume_model(data_path, X_train, X_test, Y_train, Y_test)
    JLD2.save(joinpath(data_path, "training_data.jld2"), "data", training_data)
    monomial_data = smoke_analyze_volume_epochs(data_path)
    return " ($(length(monomial_data["pre"])) epochs)"
end

function smoke_volume_dynamics_analyse()
    data_path = joinpath(SMOKE_ROOT, "volume_dynamics")
    training_data = JLD2.load(joinpath(data_path, "training_data.jld2"))["data"]

    df_acc = DataFrame(
        Epoch = 0:(length(training_data["train_acc"]) - 1),
        Train_Accuracy = training_data["train_acc"],
        Test_Accuracy = training_data["test_acc"],
        Loss = training_data["loss"],
    )
    CSV.write(joinpath(data_path, "accuracies.csv"), df_acc)

    monomial_data = JLD2.load(joinpath(data_path, "monomial_data.jld2"))["data"]
    epochs = sort(parse.(Int, filter(x -> !occursin(".", x), readdir(data_path))))
    df_mono = DataFrame(Epoch = epochs, Pre_Pruning = monomial_data["pre"], Post_Pruning = monomial_data["post"])
    CSV.write(joinpath(data_path, "monomial_counts.csv"), df_mono)

    mean_vols = Float64[]
    median_vols = Float64[]
    count_vols = Int[]
    for epoch in epochs
        edge_data = JLD2.load(joinpath(data_path, string(epoch), "edge_data.jld2"))["data"]
        finite_vols = filter(isfinite, Float64.(edge_data["lengths"]))
        push!(mean_vols, isempty(finite_vols) ? NaN : mean(finite_vols))
        push!(median_vols, isempty(finite_vols) ? NaN : median(finite_vols))
        push!(count_vols, length(finite_vols))
    end

    df_vols = DataFrame(
        Epoch = epochs,
        Mean_Finite_Volume = mean_vols,
        Median_Finite_Volume = median_vols,
        Finite_Volume_Count = count_vols,
    )
    CSV.write(joinpath(data_path, "finite_volumes_stats.csv"), df_vols)
    return " ($(nrow(df_acc)) accuracy rows)"
end

function smoke_mnist_main()
    output_dir = joinpath(SMOKE_ROOT, "mnist")
    mkpath(output_dir)

    X_train = Float32[0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0]
    y_train = onehotbatch([0, 1, 1, 0], 0:1)
    loader = DataLoader((X_train, y_train), batchsize=2, shuffle=false)

    model = Chain(Dense(2 => 2, relu), Dense(2 => 2), softmax)
    opt_state = Flux.setup(Adam(0.005), model)
    for (x, y) in loader
        _, grads = Flux.withgradient(model) do m
            crossentropy(m(x), y)
        end
        Flux.update!(opt_state, model, grads[1])
    end

    acc = mean(onecold(model(X_train)) .== onecold(y_train))
    model_state = Flux.state(model)
    jldsave(joinpath(output_dir, "model.jld2"); model_state)
    open(joinpath(output_dir, "metrics.txt"), "w") do io
        @printf(io, "Smoke Accuracy: %.2f%%\n", acc * 100)
    end
    return " (acc=$(round(acc, digits=3)))"
end

function smoke_mnist_analyse()
    output_dir = joinpath(SMOKE_ROOT, "mnist")
    model = Chain(Dense(2 => 2, relu), Dense(2 => 2), softmax)
    model_state = JLD2.load(joinpath(output_dir, "model.jld2"), "model_state")
    Flux.loadmodel!(model, model_state)

    weights, biases, thresholds = model_weights_biases_thresholds(model)
    output = mlp_to_trop(weights, biases, thresholds, quicksum=true)
    lin_regions = enum_linear_regions_rat_general(output[1]; mode=OscarMode())
    mon_count = monomial_count(output)

    analysis = Dict("trop_rep"=>output, "num_lin_region"=>length(lin_regions), "num_mon"=>mon_count)
    jldsave(joinpath(output_dir, "analysis_smoke.jld2"); analysis)
    return " ($(length(lin_regions)) regions, $mon_count monomials)"
end

function main()
    mkpath(SMOKE_ROOT)
    run_step("visualize_linear_regions/main.jl", smoke_visualize_linear_regions)
    run_step("effective_radius/main.jl", smoke_effective_radius)
    run_step("width_depth/main.jl", smoke_width_depth)
    run_step("rate_of_pruning/main.jl", smoke_rate_of_pruning)
    run_step("volume_dynamics/main.jl", smoke_volume_dynamics_main)
    run_step("volume_dynamics/analyse.jl", smoke_volume_dynamics_analyse)
    run_step("mnist/main.jl", smoke_mnist_main)
    run_step("mnist/analyse.jl", smoke_mnist_analyse)
end

main()
