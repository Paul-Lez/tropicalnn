include(joinpath(@__DIR__, "..", "experiment_setup.jl"))
const EXPERIMENT_RUNTIME = setup_experiment!()

using TropicalNN, Plots
using Printf
using DataFrames
using CSV
include("../utils.jl")

const REGION_MODE = highs_mode(EXPERIMENT_RUNTIME)
const WORKER_IDS = tropical_workers(EXPERIMENT_RUNTIME)

output_dir = "../outputs/effective_radius/"
mkpath(output_dir)

w, b, t = random_mlp([2, 4, 1])
rmap = tropicalize(w, b, t)[1]

# --- Hoffman-constant algorithm comparison -------------------------------
# Compute the Hoffman constant two ways and time each: exhaustive row-subset
# enumeration vs. the default PVZ pruning algorithm.
let warmup = tropicalize(random_mlp([2, 2, 1])...)[1]
    warmup = prune(warmup; mode=REGION_MODE)
    hoffman_constant(warmup; brute_force=true)
    hoffman_constant(warmup)
end

rmap = prune(rmap; mode=REGION_MODE)
t_exact = @elapsed hoff_exact = hoffman_constant(rmap; brute_force=true)
t_pvz   = @elapsed hoff_pvz   = hoffman_constant(rmap)

isapprox(Float64(hoff_exact), Float64(hoff_pvz); rtol=1e-6) ||
    error("Hoffman constants disagree: brute_force=$hoff_exact, pvz=$hoff_pvz")

@printf("Hoffman constant: %.10g\n", Float64(hoff_exact))
@printf("brute force: %8.4f s\n", t_exact)
@printf("PVZ:         %8.4f s  (%.2fx vs brute force)\n", t_pvz, t_exact / t_pvz)

hoffman_timings = DataFrame(
    algorithm = ["brute_force", "pvz"],
    hoffman_constant = [Float64(hoff_exact), Float64(hoff_pvz)],
    seconds = [t_exact, t_pvz],
)
CSV.write(joinpath(output_dir, "hoffman_timings.csv"), hoffman_timings)

# --- Effective radius + linear-region visualization ----------------------
er = exact_er(rmap)
regions = linear_regions(rmap; mode=REGION_MODE, workers=WORKER_IDS)

margin_limit = Float64(er * 1.2)
println(margin_limit)
fig = plot_linear_regions(regions, xlims=(-margin_limit, margin_limit), ylims=(-margin_limit, margin_limit))

sq_x = [er, -er, -er,  er, er]
sq_y = [er,  er, -er, -er, er]

plot!(fig, sq_x, sq_y, 
      color=:red, 
      linewidth=2.5, 
      linestyle=:dash, 
      label="Effective Radius Square")

savefig(fig, joinpath(output_dir, "bounding_linear_regions.png"))
