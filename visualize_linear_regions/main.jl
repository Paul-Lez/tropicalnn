include(joinpath(@__DIR__, "..", "experiment_setup.jl"))
const EXPERIMENT_RUNTIME = setup_experiment!()

using TropicalNN, Plots
include("../utils.jl")

const REGION_MODE = highs_mode(EXPERIMENT_RUNTIME)
const WORKER_IDS = tropical_workers(EXPERIMENT_RUNTIME)

output_dir = "../outputs/visualize_linear_regions/"
mkpath(output_dir)

# --- Small NN ---
weights, biases, thresholds = random_mlp([2, 4, 1])
f = mlp_to_trop(weights, biases, thresholds)[1]
regions = linear_regions(f; mode=REGION_MODE, workers=WORKER_IDS)

fig = plot_linear_regions(regions, xlims=(-5.0, 5.0), ylims=(-5.0, 5.0))
savefig(fig, joinpath(output_dir, "small_nn.png"))

# --- Large NN ---
weights, biases, thresholds = random_mlp([2, 12, 1])
f = mlp_to_trop(weights, biases, thresholds)[1]
regions = linear_regions(f; mode=REGION_MODE, workers=WORKER_IDS)

fig = plot_linear_regions(regions, xlims=(-5.0, 5.0), ylims=(-5.0, 5.0))
savefig(fig, joinpath(output_dir, "large_nn.png"))

# # --- Deep NN ---
# weights, biases, thresholds = random_mlp([2, 5, 5, 1])
# f = mlp_to_trop(weights, biases, thresholds)[1]
# regions = linear_regions(f; mode=HiGHSMode())
# println("Deep NN [2, 5, 5, 1]: $(length(regions)) linear regions")

# fig = plot_linear_regions(regions, xlims=(-5.0, 5.0), ylims=(-5.0, 5.0))
# savefig(fig, joinpath(output_dir, "deep_nn.png"))
