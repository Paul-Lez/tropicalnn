include("../paper/experiments/load_packages.jl")
include("../paper/experiments/paper_computations.jl")
using JSON3
using Dates

# results are saved in a file named `experiment_5_1_run_DATE&TIME.json`
results_file = "experiment_5_1_run_" *  string(Dates.now()) * ".json"

# experimental parameters
n_samples = 10
architectures = [   [2, 4, 4, 1], 
                    [2, 8, 1], 
                    [10, 4, 1],
                    [10, 2, 2, 1]
                ] 

t1 = time()
exp3_mon, compute_times = monomial_counting(architectures, n_samples, "")
t2 = time()
println("Experiment completed in ", t2 - t1, " seconds")
experiment = Dict("Monomials" => exp3_mon, "Compute times" => compute_times, "input"=> architectures, "time" => t2 - t1)

open(results_file, "w") do io
    JSON3.pretty(io, experiment)
end 