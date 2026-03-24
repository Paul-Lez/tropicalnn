using TropicalNN
using DataFrames
using CSV
using Logging

global_logger(SimpleLogger(stderr, Logging.Error))

# Helper to get monomial counts straight from generated dimensions
function count_monomials(dims)
    w, b, t = random_mlp(dims)
    
    f_pre = mlp_to_trop(w, b, t)[1]
    
    return monomial_count(f_pre)
end

function run_experiments()
    num_trials = 3
    mkpath("outputs/width_depth")

    widths = [2, 3, 4, 5, 6, 7, 8] 
    width_results = DataFrame(Width=Int[], Pre_Avg=Float64[])
    
    println("--- Starting OneLayer Sweep ---")
    for w in widths
        println("Processing Width: $w")
        sum_pre = 0.0
        
        for _ in 1:num_trials
            pre = count_monomials([2, w, 1])
            sum_pre += pre
        end
        
        push!(width_results, (w, sum_pre / num_trials))
    end
    CSV.write("outputs/width_depth/sweep_onelayer.csv", width_results)
    

    twolayer_results = DataFrame(Depth=Int[], Pre_Avg=Float64[])
    println("\n--- Starting TwoLayer Sweep ---")
    for w in widths
        println("Processing Width: $w")
        sum_pre = 0.0
        
        dims = vcat([2, 2, w, 1])
        
        for _ in 1:num_trials
            pre = count_monomials(dims)
            sum_pre += pre
        end
        
        push!(twolayer_results, (w, sum_pre / num_trials))
    end
    CSV.write("outputs/width_depth/sweep_twolayer.csv", twolayer_results)
end

run_experiments()