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
rmap = mlp_to_trop(w, b, t)[1]

# --- Hoffman-constant algorithm comparison -------------------------------
# Compute the Hoffman constant two ways and time each: the brute-force
# exact_hoff (tests every row subset) vs. the PVZ pruning algorithm (pvz_hoff).
let warmup = mlp_to_trop(random_mlp([2, 2, 1])...)[1]
    exact_hoff(warmup; mode=REGION_MODE)
    pvz_hoff(warmup; mode=REGION_MODE)
end

t_exact = @elapsed hoff_exact = exact_hoff(rmap; mode=REGION_MODE)
t_pvz   = @elapsed hoff_pvz   = pvz_hoff(rmap; mode=REGION_MODE)

isapprox(Float64(hoff_exact), Float64(hoff_pvz); rtol=1e-6) ||
    error("Hoffman constants disagree: exact_hoff=$hoff_exact, pvz_hoff=$hoff_pvz")

@printf("Hoffman constant: %.10g\n", Float64(hoff_exact))
@printf("exact_hoff: %8.4f s\n", t_exact)
@printf("pvz_hoff:   %8.4f s  (%.2fx vs exact_hoff)\n", t_pvz, t_exact / t_pvz)

hoffman_timings = DataFrame(
    algorithm = ["exact_hoff", "pvz_hoff"],
    hoffman_constant = [Float64(hoff_exact), Float64(hoff_pvz)],
    seconds = [t_exact, t_pvz],
)
CSV.write(joinpath(output_dir, "hoffman_timings.csv"), hoffman_timings)

# --- Effective radius + linear-region visualization ----------------------
er = exact_er(rmap; mode=REGION_MODE)
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
