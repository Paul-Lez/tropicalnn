include(joinpath(@__DIR__, "..", "experiment_setup.jl"))
const EXPERIMENT_RUNTIME = setup_experiment!()

using CSV
using DataFrames
import Flux
using Flux: DataLoader
import Graphs
using JLD2
using MLDatasets
using Plots
using Printf
using Statistics
using TropicalNN

const REGION_MODE = highs_mode(EXPERIMENT_RUNTIME)
const WORKER_IDS = tropical_workers(EXPERIMENT_RUNTIME)

include("../utils.jl")
include("../mnist/models.jl")
using .MNISTModels

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

    weights, biases, thresholds = random_mlp([2, 3, 1])
    f = tropicalize(weights, biases, thresholds)[1]
    
    counts = Int[]
    for (name, mode) in (("oscar", OscarMode()), ("highs", REGION_MODE))
        regions = linear_regions(f; mode=mode, workers=WORKER_IDS)
        fig = plot_linear_regions(regions, xlims=(-2.0, 2.0), ylims=(-2.0, 2.0))
        savefig(fig, joinpath(output_dir, "tiny_nn_$name.png"))
        push!(counts, length(regions))
    end
    return " (oscar=$(counts[1]), highs=$(counts[2]) regions)"
end

function smoke_effective_radius()
    output_dir = joinpath(SMOKE_ROOT, "effective_radius")
    mkpath(output_dir)

    weights, biases, thresholds = random_mlp([2, 3, 1])
    rmap = tropicalize(weights, biases, thresholds)[1]
    rmap_oscar = TropicalNN.prune(rmap; mode=OscarMode())
    rmap_highs = TropicalNN.prune(rmap; mode=REGION_MODE)

    # The real experiment times the two Hoffman-constant algorithms against each
    # other (brute-force enumeration vs. PVZ pruning) and derives the
    # effective radius from the constant. Exercise both algorithms and check they
    # agree, then compute exact_er under both LP backends.
    hoff_exact = hoffman_constant(rmap_highs; brute_force=true)
    hoff_pvz   = hoffman_constant(rmap_highs)
    @assert isapprox(Float64(hoff_exact), Float64(hoff_pvz); rtol=1e-6) "brute-force vs PVZ Hoffman constants disagree: $hoff_exact vs $hoff_pvz"

    er_oscar = exact_er(rmap_oscar)
    er_highs = exact_er(rmap_highs)
    er_upper = upper_er(rmap_highs)

    regions = linear_regions(rmap_highs; mode=REGION_MODE, workers=WORKER_IDS)
    exact_fig = plot_radius_bound(regions, er_highs)
    savefig(exact_fig, joinpath(output_dir, "bounding_linear_regions.png"))

    upper_fig = plot_radius_bound(regions, er_upper)
    savefig(upper_fig, joinpath(output_dir, "bounding_linear_regions_upper_er.png"))
    return " (er_oscar=$(Float64(er_oscar)), er_highs=$(Float64(er_highs)), er_upper=$(Float64(er_upper)))"
end

function smoke_width_depth()
    output_dir = joinpath(SMOKE_ROOT, "width_depth")
    mkpath(output_dir)

    weights, biases, _ = random_mlp([2, 2, 1]; symbolic = false)
    relu_network = NeuralNetwork(
        AffineLayer(weights[1], biases[1]),
        ActivationLayer(relu(Float64), 2),
        AffineLayer(weights[2], biases[2]),
    )
    maxout_network = random_maxout_network([2, 2, 1], 2, Float64)
    @assert relu_network isa NeuralNetwork{Float64}
    @assert maxout_network isa NeuralNetwork{Float64}

    relu_regions = linear_regions(relu_network; mode = REGION_MODE, workers = WORKER_IDS)
    maxout_regions = linear_regions(maxout_network; mode = REGION_MODE, workers = WORKER_IDS)
    results = DataFrame(
        Network = ["relu", "maxout"],
        NumRegions = length.((relu_regions, maxout_regions)),
        Algorithm = fill("HiGHS", 2),
        Encoding = fill("Float64", 2),
    )
    CSV.write(joinpath(output_dir, "linear_regions.csv"), results)

    # CSV infers a Missing-only column from a ReLU-only partial checkpoint.
    # Loading through the declared schema must preserve room for later maxout rows.
    partial_path = joinpath(output_dir, "partial_checkpoint.csv")
    CSV.write(partial_path, DataFrame(Network = ["relu"], Pieces = [missing]))
    partial_schema = DataFrame(
        Network = String[],
        Pieces = Union{Missing, Int}[],
    )
    resumed = load_typed_csv(partial_path, partial_schema)
    @assert eltype(resumed.Pieces) == Union{Missing, Int}
    push!(resumed, ("maxout", 2))
    return " (relu=$(length(relu_regions)), maxout=$(length(maxout_regions)) regions)"
end

function smoke_get_monomial_counts(model)
    weights, biases, thresholds = model_weights_biases_thresholds(model)
    f_pre = tropicalize(weights, biases, thresholds)[1]
    f_post = TropicalNN.prune(f_pre; mode=REGION_MODE, workers=WORKER_IDS)
    return monomial_count(f_pre), monomial_count(f_post)
end

function smoke_rate_of_pruning()
    output_dir = joinpath(SMOKE_ROOT, "rate_of_pruning")
    mkpath(output_dir)

    X_train, _, Y_train, _ = tiny_binary_data()
    X_train_mat = Matrix(X_train')
    Y_train_mat = reshape(Y_train, 1, :)

    weights, biases, _ = random_mlp([2, 1, 1])
    model = Flux.Chain(
        Flux.Dense(weights[1], biases[1], Flux.relu),
        Flux.Dense(weights[2], biases[2], identity),
        Flux.σ,
    )
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
    model = Flux.Chain(
        Flux.Dense(weights[1], biases[1], Flux.relu),
        Flux.Dense(weights[2], biases[2], identity),
        Flux.σ,
    )

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

        f_pre = tropicalize(weights, biases, thresholds)[1]
        f_post = TropicalNN.prune(f_pre; mode=REGION_MODE, workers=WORKER_IDS)
        graph = TropicalNN.get_graph(f_post; mode=REGION_MODE)
        edge_data = Dict(
            "gradients"=>edge_directions(f_post; mode=REGION_MODE)["full"],
            "lengths"=>edge_lengths(f_post; mode=REGION_MODE)["full"],
        )

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
        graph = JLD2.load(joinpath(data_path, string(epoch), "graph.jld2"))["graph"]
        finite_vols = Float64[]
        for vertex in Graphs.vertices(graph)
            region_volume = sum(Float64.(graph[vertex]["volume"]))
            isfinite(region_volume) && push!(finite_vols, region_volume)
        end
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
    y_train = Flux.onehotbatch([0, 1, 1, 0], 0:1)
    loader = DataLoader((X_train, y_train), batchsize=2, shuffle=false)
    requested_specs = experiment_specs()
    observed_specs = [
        (spec.activation, spec.widths, spec.pieces)
        for spec in requested_specs
    ]
    expected_specs = vcat(
        [(:relu, [width], 1) for width in 4:8],
        [(:maxout, [width], 2) for width in 4:8],
        [(:relu, [width, width], 1) for width in 4:8],
    )
    @assert observed_specs == expected_specs "MNIST experiment architecture matrix changed"

    smoke_specs = (
        ExperimentSpec("smoke_relu", :relu, [2]),
        ExperimentSpec("smoke_maxout", :maxout, [2]; pieces = 2),
        ExperimentSpec("smoke_relu_depth2", :relu, [2, 2]),
    )
    accuracies = Float64[]

    for spec in smoke_specs
        model = build_model(spec; input_dimension = 2, output_dimension = 2)
        opt_state = Flux.setup(Flux.Adam(0.005), model)
        for (x, y) in loader
            _, grads = Flux.withgradient(model) do m
                Flux.crossentropy(m(x), y)
            end
            Flux.update!(opt_state, model, grads[1])
        end

        push!(accuracies, mean(Flux.onecold(model(X_train)) .== Flux.onecold(y_train)))
        model_state = Flux.state(model)
        jldsave(joinpath(output_dir, "$(spec.id).jld2"); model_state)
    end
    return " (relu/maxout accuracies=$(round.(accuracies; digits = 3)))"
end

function smoke_mnist_analyse()
    output_dir = joinpath(SMOKE_ROOT, "mnist")
    smoke_specs = (
        ExperimentSpec("smoke_relu", :relu, [2]),
        ExperimentSpec("smoke_maxout", :maxout, [2]; pieces = 2),
        ExperimentSpec("smoke_relu_depth2", :relu, [2, 2]),
    )
    region_counts = Int[]
    for spec in smoke_specs
        model = build_model(spec; input_dimension = 2, output_dimension = 2)
        model_state = JLD2.load(joinpath(output_dir, "$(spec.id).jld2"), "model_state")
        Flux.loadmodel!(model, model_state)

        network = model_to_tropical(model, spec)
        @assert network isa NeuralNetwork{Float64}
        if spec.activation == :maxout
            point = Float32[0.25, -0.75]
            hidden_output = model[1](point)
            flux_logits = model[2](hidden_output)
            tropical_output = tropicalize(network; quicksum = true, dedup = true)
            tropical_logits = TropicalNN.TropicalNumbers.content.(
                TropicalNN.evaluate(tropical_output, Float64.(point))
            )
            @assert isapprox(tropical_logits, Float64.(flux_logits); atol = 1e-6)
        end
        regions = linear_regions(network; mode = REGION_MODE, workers = WORKER_IDS)
        push!(region_counts, length(regions))
    end
    analysis = Dict(
        "num_lin_regions" => region_counts,
        "algorithm" => "HiGHS",
        "encoding" => "Float64",
    )
    jldsave(joinpath(output_dir, "analysis_smoke.jld2"); analysis)
    return " (relu/maxout regions=$region_counts, Float64/HiGHS)"
end

# Exercise the distributed worker path used by the real experiments. If the
# smoke test was launched with --processes N (so WORKER_IDS is populated) every
# step above already ran distributed; this step additionally checks that the
# parallel results agree with the serial ones on a net large enough to actually
# chunk across workers, spinning up its own workers when none were requested.
function smoke_distributed()
    weights, biases, thresholds = random_mlp([2, 4, 1])
    f = tropicalize(weights, biases, thresholds)[1]

    serial_regions = length(linear_regions(f; mode=REGION_MODE))
    serial_reduced = monomial_count(TropicalNN.prune(f; mode=REGION_MODE))

    owned_workers = Int[]
    worker_pool = WORKER_IDS
    if worker_pool === nothing
        owned_workers = Distributed.addprocs(2; exeflags="--project=$(Base.active_project())")
        _configure_worker_environment!(owned_workers, EXPERIMENT_RUNTIME.highs_threads)
        _load_tropicalnn_on_workers!(owned_workers)
        worker_pool = Distributed.WorkerPool(owned_workers)
    end

    try
        par_regions = length(linear_regions(f; mode=REGION_MODE, workers=worker_pool))
        par_reduced = monomial_count(TropicalNN.prune(f; mode=REGION_MODE, workers=worker_pool))
        @assert par_regions == serial_regions "distributed linear_regions ($par_regions) != serial ($serial_regions)"
        @assert par_reduced == serial_reduced "distributed prune ($par_reduced) != serial ($serial_reduced)"
        return " ($(length(Distributed.workers(worker_pool))) workers, regions=$par_regions, reduced monomials=$par_reduced)"
    finally
        isempty(owned_workers) || Distributed.rmprocs(owned_workers)
    end
end

function main()
    mkpath(SMOKE_ROOT)
    run_step("visualize_linear_regions/main.jl", smoke_visualize_linear_regions)
    run_step("effective_radius/main.jl", smoke_effective_radius)
    run_step("width_depth/linear_regions.jl", smoke_width_depth)
    run_step("rate_of_pruning/main.jl", smoke_rate_of_pruning)
    run_step("volume_dynamics/main.jl", smoke_volume_dynamics_main)
    run_step("volume_dynamics/analyse.jl", smoke_volume_dynamics_analyse)
    run_step("mnist/main.jl", smoke_mnist_main)
    run_step("mnist/analyse.jl", smoke_mnist_analyse)
    run_step("distributed workers", smoke_distributed)
end

main()
