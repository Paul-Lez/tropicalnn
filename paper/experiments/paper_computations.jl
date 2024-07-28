using Pkg 
using Oscar
using Combinatorics
using Plots
using JLD2
include("../../src/rat_maps.jl")
include("../../src/linear_regions.jl")
include("../../src/mlp_to_trop.jl")
include("../../src/mlp_to_trop_with_elim.jl")

# This file contains the tropical geometry experiments for the paper 

"""
Experiment 1: compute number of linear regions as the number of monomials varies.
"""
function rational_map_linear_region_computations(n_variables, n_terms, n_samples)
    output = Dict()
    compute_times = Dict()
    for i in Base.eachindex(n_terms)
        println("Linear regions computation for ", n_terms[i], " variables \n")
        sample_times = []
        sample_n_regions = []
        for s in 1:n_samples 
            println("Sample ", s, " out of ", n_samples)
            t1 = time()
            # Pick random coefficients for numerator and denominator
            c_f = [QQ(Rational(j)) for j in rand(Float64, n_terms[i])]
            c_g = [QQ(Rational(j)) for j in rand(Float64, n_terms[i])]
            # initialise exponent arrays
            exp_f = Vector{Vector{Rational{Int}}}()
            exp_g = Vector{Vector{Rational{Int}}}()
            sizehint!(exp_f, n_terms[i])
            sizehint!(exp_g, n_terms[i])
            # pick random exponents
            for j in 1:n_terms[i]
                new_exp_f = Vector{Rational{Int}}([Rational(j) for j in rand(Float64, n_variables)])
                new_exp_g = Vector{Rational{Int}}([Rational(j) for j in rand(Float64, n_variables)])
                push!(exp_f, new_exp_f)
                push!(exp_g, new_exp_g)
            end
            f = TropicalPuiseuxPoly(c_f, exp_f)
            g = TropicalPuiseuxPoly(c_g, exp_g)
            # compute the number of linear regions of the rational function
            n_reg = 0
            try 
                n_reg = length(enum_linear_regions_rat(f, g, true))
            catch e
                println("Oscar error")
            end 
            t2 = time()
            # store the runtime and number of linear regions 
            push!(sample_times, t2-t1)
            push!(sample_n_regions, n_reg)
            println(" ")
        end 
        # compute averages 
        sample_average_time = sum(sample_times) / n_samples
        sample_average_region = sum(sample_n_regions) / n_samples
        # store information in output dictionaries
        compute_times["Computation " * string(i)] = (sample_average_time, sample_times)
        output["Computation " * string(i)] = (sample_average_region, sample_n_regions)
    end 
    return output, compute_times
end 

"""
Experiment 2: Compute number of monomials that appear in the tropical Puiseux rational expression of a neural network and vary the architecture.
"""
function monomial_counting(architectures, n_samples, save_file)
    # this dictionary will store the number of monomials for each architecture
    n_monomials = Dict()
    runtimes = Dict()
    for i in Base.eachindex(architectures)
        # these arrays store outcome of computation and runtime for each individual computation 
        sample_times = []
        sample_results = []
        println("Currently working on architecture ", architectures[i])
        for j in 1:n_samples
            println("Sample ", j, " out of ", n_samples)
            save_file_name = save_file*"tropical_nn_architecture_" * string(i) * "_sample_" * string(j) * ".jld2"
            t1 = time()
            # pick a random neural network with a given architecture 
            weights, bias, thresholds = random_mlp(architectures[i], false)
            # compute the corresponding array of tropical Puiseux rational maps
            trop = mlp_to_trop(weights, bias, thresholds)
            # count the number of monomials that appear there, i.e. sum of monomials in numeral and denominator over all possible entries of the array
            n_mon = sum([length(i.den.exp)+length(i.num.exp) for i in trop])
            t2 = time()
            println("Found ", n_mon, " monomials.")
            push!(sample_results, n_mon) 
            push!(sample_times, t2-t1)
            # save tropical rational function in file 
            save_object(save_file_name, trop)
        end 
        # compute averages 
        average_n_monomial = sum(sample_results) / n_samples
        average_runtime = sum(sample_times) / n_samples 
        # store data in output dictionaries
        sample_output = Dict("Average number of monomials" => average_n_monomial, "Individual samples" => sample_results)
        sample_runtime = Dict("Average runtime" => average_runtime, "Individual sample runtimes" => sample_times)
        # add the sample dictionaries to the experiment output dictionaries'
        n_monomials["Architecture " * string(i)] = sample_output
        runtimes["Architecture " * string(i)] = sample_runtime
    end
    return n_monomials, runtimes
end 

"""
Experiment 3: Compute the number of linear regions of a neural network as architecture varies
"""
function untrained_linear_region_computations(architectures, n_samples, save_file)
    n_regions = Dict()
    n_monomials = Dict()
    runtimes = Dict()
    for i in Base.eachindex(architectures)
        sample_times = []
        sample_results_n_mon = []
        sample_results_n_reg = []
        for j in 1:n_samples
            println("Currently working on architecture ", architectures[i])
            #save_file*
            save_file_name_trop = save_file*"architecture_" * string(i) * "_sample_" * string(j) * "_trop.jld2"
            save_file_name_lin = save_file*"architecture_" * string(i) * "_sample_" * string(j) * "_lin.jld2"
            t1 = time()
            # with a random neural network with a given architecture 
            weights, bias, thresholds = random_mlp(architectures[i], false)
            # compute the tropical Puiseux rational map
            trop = mlp_to_trop_with_quasi_elim(weights, bias, thresholds)[1]
            # count the number of linear regions of the network
            reg = []
            try 
                reg = enum_linear_regions_rat(trop.num, trop.den)
            catch e 
                println("Oscar error ", e)
            end 
            n_reg = length(reg)
            n_mon = length(trop.den.exp)+length(trop.num.exp)
            t2 = time()
            println("Found ", n_mon, " monomials, and ", n_reg, " linear regions.")
            push!(sample_results_n_mon, n_mon) 
            push!(sample_times, t2-t1)
            push!(sample_results_n_reg, n_reg)
            # save tropical rational function in file 
            save_object(save_file_name_trop, trop)
            save_object(save_file_name_lin, reg)
        end 
        # compute averages 
        average_n_monomial = sum(sample_results_n_mon) / n_samples
        average_runtime = sum(sample_times) / n_samples 
        average_n_regions = sum(sample_results_n_reg) / n_samples
        # store data in output dictionaries
        sample_output_n_mon = Dict("Average number of monomials" => average_n_monomial, "Individual samples" => sample_results_n_mon)
        sample_output_n_reg =  Dict("Average number of linear regions" => average_n_regions, "Individual samples" => sample_results_n_reg)
        sample_runtime = Dict("Average runtime" => average_runtime, "Individual sample runtimes" => sample_times)
        # add the sample dictionaries to the experiment output dictionaries'
        n_monomials["Architecture " * string(i)] = sample_output_n_mon
        runtimes["Architecture " * string(i)] = sample_runtime
        n_regions["Architecture " * string(i)] = sample_output_n_reg 
    end
    return n_regions, n_monomials, runtimes
end 
