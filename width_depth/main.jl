include(joinpath(@__DIR__, "..", "experiment_setup.jl"))
const EXPERIMENT_RUNTIME = setup_experiment!()

using TropicalNN
using DataFrames
using CSV
using Logging

const REGION_MODE = highs_mode(EXPERIMENT_RUNTIME)
const WORKER_IDS = tropical_workers(EXPERIMENT_RUNTIME)

global_logger(SimpleLogger(stderr, Logging.Error))

# For one random network, compute the unpruned tropical representation and the
# layerwise-pruned one from the same weights.
function count_monomials(dims)
    w, b, t = random_mlp(dims, symbolic=false)

    t_unpruned = @elapsed f_unpruned = tropicalize(w, b, t, quicksum=true, dedup=true)[1]
    t_pruned = @elapsed f_pruned = tropicalize(w, b, t, quicksum=true, dedup=true,
        prune=true, elim_mode=REGION_MODE, workers=WORKER_IDS)[1]

    return monomial_count(f_unpruned), monomial_count(f_pruned), t_unpruned, t_pruned
end

function run_experiments()
    num_trials = 30
    mkpath("outputs/width_depth")

    widths = [2, 3, 4, 5, 6, 7, 8]
    width_results = DataFrame(Width=Int[], Unpruned_Avg=Float64[], Pruned_Avg=Float64[],
        Unpruned_Time_Avg=Float64[], Pruned_Time_Avg=Float64[])

    println("--- Starting OneLayer Sweep ---")
    for w in widths
        println("Processing Width: $w")
        sum_unpruned = 0.0
        sum_pruned = 0.0
        sum_t_unpruned = 0.0
        sum_t_pruned = 0.0

        for _ in 1:num_trials
            unpruned, pruned, t_unpruned, t_pruned = count_monomials([2, w + 2, 1])
            sum_unpruned += unpruned
            sum_pruned += pruned
            sum_t_unpruned += t_unpruned
            sum_t_pruned += t_pruned
        end

        push!(width_results, (w, sum_unpruned / num_trials, sum_pruned / num_trials,
            sum_t_unpruned / num_trials, sum_t_pruned / num_trials))
    end
    CSV.write("outputs/width_depth/sweep_onelayer.csv", width_results)


    twolayer_results = DataFrame(Depth=Int[], Unpruned_Avg=Float64[], Pruned_Avg=Float64[],
        Unpruned_Time_Avg=Float64[], Pruned_Time_Avg=Float64[])

    println("\n--- Starting TwoLayer Sweep ---")
    for w in widths
        println("Processing Width: $w")
        sum_unpruned = 0.0
        sum_pruned = 0.0
        sum_t_unpruned = 0.0
        sum_t_pruned = 0.0

        dims = vcat([2, w, 2, 1])

        for _ in 1:num_trials
            unpruned, pruned, t_unpruned, t_pruned = count_monomials(dims)
            sum_unpruned += unpruned
            sum_pruned += pruned
            sum_t_unpruned += t_unpruned
            sum_t_pruned += t_pruned
        end

        push!(twolayer_results, (w, sum_unpruned / num_trials, sum_pruned / num_trials,
            sum_t_unpruned / num_trials, sum_t_pruned / num_trials))
    end
    CSV.write("outputs/width_depth/sweep_twolayer.csv", twolayer_results)
end

run_experiments()
