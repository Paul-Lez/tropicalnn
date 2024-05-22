include("../paper/experiments/paper_computations.jl")

# results are saved in a file named `experiment_5_1_run_DATE&TIME.json`
results_file = "experiment_5_1_run_" *  string(Dates.now()) * ".json"

# experimental parameters
n_variables = 3
n_terms = [20, 50, 100, 200, 350, 500, 800, 1000]
n_samples = 12

t1 = time()
#lauch the experiment
exp1, compute_times = rational_map_linear_region_computations(n_variables, n_terms, n_samples) 
t2 = time()
println("Experiment 1 completed in ", t2 - t1, " seconds")
experiment = Dict("output" => exp1, "input"=> n_terms, "time" => t2 - t1, "compute times" => compute_times)

open(results_file, "w") do io
    JSON3.pretty(io, experiment)
end 



