include("../paper/experiments/paper_computations.jl")

# results are saved in a file named `experiment_5_1_run_DATE&TIME.json`
results_file = "experiment_5_1_run_" *  string(Dates.now()) * ".json"

# experimental parameters
n_samples = 100
architectures = [[2, 2, 1], 
                [3, 3, 1], 
                [4, 4, 2, 1], 
                [4, 3, 3, 2, 1], 
                [4, 4, 3, 2, 1], 
                [4, 4, 4, 2, 1]] 

t1 = time()
exp3_lin, exp3_mon, compute_times = untrained_linear_region_computations(architectures, n_samples, "data/computation_objects/")
t2 = time()
println("Experiment 3 completed in ", t2 - t1, " seconds")
experiment3 = Dict("Linear regions" => exp3_lin, "Monomials" => exp3_mon, "Compute times" => compute_times, "input"=> architectures, "time" => t2 - t1)
results_dict["experiment3"] = experiment3