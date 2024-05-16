#include("load_packages.jl")
include("paper_computations.jl")
using JSON3
using Dates

#TODO: add parser to allow custom inputs from the command line


# parameters for the experiements
n_variables = 4
n_terms = [20] # , 50, 100, 200, 350, 500, 800, 1000
n_samples = 4
architectures = [[2, 2, 1], 
                [3, 3, 1], 
                [4, 4, 2, 1], 
                [4, 3, 3, 2, 1], 
                [4, 4, 3, 2, 1], 
                [4, 4, 4, 2, 1]] 


# modify these values to select which experiments to run. 
run1 = false
run2 = true 
run3 = false
run4 = false

# results of each experiment will be saved in a .json log file. By default the file name is "experiment_run_DATE_AND_TIME.json"
results_file = "results/experiment_run_" *  string(Dates.now()) * ".json"
results_dict = Dict()

if run1
    t1 = time()
    #lauch first experiment 
    exp1, compute_times = rational_map_linear_region_computations(n_variables, n_terms, n_samples) 
    t2 = time()
    println("Experiment 1 completed in ", t2 - t1, " seconds")
    experiment1 = Dict("output" => exp1, "input"=> n_terms, "time" => t2 - t1, "compute times" => compute_times)
    results_dict["experiment1"] = experiment1
end 

if run2 
    t1 = time()
    exp2, compute_times = monomial_counting(architectures, n_samples)
    t2 = time()
    println("Experiment 2 completed in ", t2 - t1, " seconds")
    experiment2 = Dict("output" => exp2, "input"=> architectures, "time" => t2 - t1, "compute times" => compute_times)
    results_dict["experiment2"] = experiment2
end 
    
if run3 
    t1 = time()
    exp3 = untrained_linear_region_computations(architectures)
    t2 = time()
    println("Experiment 3 completed in ", t2 - t1, " seconds")
    experiment3 = Dict("output" => exp3, "input"=> architectures, "time" => t2 - t1)
    results_dict["experiment3"] = experiment3
end

#TODO: add experiment 4
open(results_file, "w") do io
    JSON3.pretty(io, results_dict)
end 
