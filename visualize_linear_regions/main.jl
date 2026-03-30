using TropicalNN, Plots
include("../utils.jl")

output_dir = "../outputs/visualize_linear_regions/"
mkpath(output_dir)

# --- Small NN ---
weights, biases, thresholds = random_mlp([2, 4, 1])
f = mlp_to_trop(weights, biases, thresholds)[1]
linear_regions = enum_linear_regions_rat(f)

fig = plot_linear_regions(linear_regions, xlims=(-5.0, 5.0), ylims=(-5.0, 5.0))
savefig(fig, joinpath(output_dir, "small_nn.png"))

# --- Large NN ---
weights, biases, thresholds = random_mlp([2, 12, 1])
f = mlp_to_trop(weights, biases, thresholds)[1]
linear_regions = enum_linear_regions_rat(f)

fig = plot_linear_regions(linear_regions, xlims=(-5.0, 5.0), ylims=(-5.0, 5.0))
savefig(fig, joinpath(output_dir, "large_nn.png"))