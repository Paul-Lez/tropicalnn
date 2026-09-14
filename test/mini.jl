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
import Random
using Statistics
using TropicalNN

const REGION_MODE = highs_mode(EXPERIMENT_RUNTIME)
const WORKER_IDS = tropical_workers(EXPERIMENT_RUNTIME)

include("../utils.jl")
include("../mnist/models.jl")
using .MNISTModels
include("../volume_dynamics/experiment.jl")
import .VolumeDynamicsExperiment

const SMOKE_ROOT = joinpath(@__DIR__, "..", "outputs", "smoke")
const VOLUME_SMOKE_PATH = Ref{String}()

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

function smoke_volume_dynamics_main()
    data_path = mktempdir(SMOKE_ROOT; prefix = "volume_dynamics_")
    VOLUME_SMOKE_PATH[] = data_path
    dataset = VolumeDynamicsExperiment.generate_dataset(
        VolumeDynamicsExperiment.SpiralGenerator(noise = 0.01, turns = 0.5),
        Random.MersenneTwister(101);
        train_size = 20,
        validation_size = 8,
        test_size = 8,
    )
    VolumeDynamicsExperiment.save_dataset(joinpath(data_path, "dataset.jld2"), dataset)
    model = VolumeDynamicsExperiment.build_model(Random.MersenneTwister(102), 2)
    @assert eltype(model[1].weight) == Float64
    training_data, _ = VolumeDynamicsExperiment.train_model!(
        model,
        dataset,
        data_path;
        batch_size = 4,
        learning_rate = 1e-3,
        weight_decay = 1e-4,
        max_steps = 1,
        checkpoint_every = 1,
        rng = Random.MersenneTwister(103),
    )
    @assert training_data["step"] == [0, 1]
    parameters = JLD2.load(joinpath(
        data_path, "checkpoints", "00000000", "parameters.jld2"
    ))
    @assert eltype(parameters["weights"][1]) == Rational{BigInt}
    monomial_data = VolumeDynamicsExperiment.analyze_checkpoints(
        data_path;
        mode = REGION_MODE,
        workers = WORKER_IDS,
    )
    return " ($(length(monomial_data["step"])) checkpoints)"
end

function smoke_volume_dynamics_analyse()
    data_path = VOLUME_SMOKE_PATH[]
    training_data = JLD2.load(joinpath(data_path, "training_data.jld2"))["training_data"]
    final_metrics = JLD2.load(joinpath(data_path, "final_metrics.jld2"))["final_metrics"]
    graph = JLD2.load(joinpath(
        data_path, "checkpoints", "00000000", "graph.jld2"
    ))["graph"]
    @assert training_data["step"] == [0, 1]
    @assert final_metrics["step"] == 1
    region_volumes = [
        sum(Float64.(graph[vertex]["volume"])) for vertex in Graphs.vertices(graph)
    ]
    @assert any(isinf, region_volumes)
    @assert all(volume -> isfinite(volume) || isinf(volume), region_volumes)
    return " ($(length(training_data["step"])) metric rows)"
end

function smoke_mnist_main()
    output_dir = joinpath(SMOKE_ROOT, "mnist")
    mkpath(output_dir)

    X_train = Float32[0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0]
    y_train = Flux.onehotbatch([0, 1, 1, 0], 0:1)
    loader = DataLoader((X_train, y_train), batchsize=2, shuffle=false)
    requested_specs = experiment_specs()
    @assert DEFAULT_NUM_SAMPLES == 30 "MNIST default sample count changed"
    @assert endswith(model_path(output_dir, first(requested_specs)), "relu_d1_w4.jld2")
    @assert endswith(
        model_path(output_dir, first(requested_specs), 2),
        "relu_d1_w4_sample02.jld2",
    )
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
