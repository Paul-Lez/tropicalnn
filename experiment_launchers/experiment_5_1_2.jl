include("../paper/experiments/load_packages.jl")
include("../paper/experiments/paper_computations.jl")
using JSON3
using Dates

# results are saved in a file named `experiment_5_1_run_DATE&TIME.json`
results_file = "experiment_5_1_run_" *  string(Dates.now()) * ".json"

##### experimental parameters ####
n_variables = 4  # Number of variables occuring in the Puiseux rational functions
n_terms = [20, 50, 100, 200, 350, 500, 800, 1000] # Number of monomials
n_samples = 4   # Number of samples used in the computation

t1 = time()
#lauch the experiment
exp1, compute_times = rational_map_linear_region_computations(n_variables, n_terms, n_samples) 
t2 = time()
println("Experiment completed in ", t2 - t1, " seconds")
experiment = Dict("Number of linear regions" => exp1, "input" => n_terms, "time" => t2 - t1, "compute times" => compute_times)

open(results_file, "w") do io
    JSON3.pretty(io, experiment)
end 