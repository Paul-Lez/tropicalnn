using Pkg 
using Oscar
using Combinatorics
using Plots
using JLD2
include("rat_maps.jl")
include("linear_regions.jl")
include("mlp_to_trop.jl")

# This file contains the tropical geometry experiments for the paper 

"""
Experiment 1: compute number of linear regions as the number of monomials varies.
"""
function rational_map_linear_region_computations(n_variables, n_terms, n_samples)
    output = []
    for i in Base.eachindex(n_terms)
        println("Linear regions computation for ", n_terms[i], " variables \n")
        n_lin = 0
        for s in 1:n_samples 
            println("Computation for sample ", s)
            c_f = [QQ(Rational(j)) for j in rand(Float64, n_terms[i])]
            c_g = [QQ(Rational(j)) for j in rand(Float64, n_terms[i])]
            exp_f = []
            exp_g = []
            for j in 1:n_terms[i]
                push!(exp_f, [QQ(Rational(j)) for j in rand(Float64, n_variables)])
                push!(exp_g, [QQ(Rational(j)) for j in rand(Float64, n_variables)])
            end
            #println(exp_f)
            f = TropicalPuiseuxPoly(c_f, exp_f)
            g = TropicalPuiseuxPoly(c_g, exp_g)
            n_lin += length(enum_linear_regions_rat(f, g, true))
            println(" ")
        end 
        push!(output, n_lin / n_samples)
    end 
    return output
    #JLD2.save_object("output_array.jld2", output)
end 

"""
Experiment 2: Compute number of monomials that appear in the tropical Puiseux rational expression of a neural network and vary the architecture.
"""
function monomial_counting(architectures)
    # this array will store the number of monomials for each architecture
    n_monomials = zeros(length(architectures))
    for i in Base.eachindex(architectures)
        # with a random neural network with a given architecture 
        weights, bias, thresholds = random_mlp(architectures[i], false)
        # compute the corresponding array of tropical Puiseux rational maps
        trop = mlp_to_trop(weights, bias, thresholds)
        # count the number of monomials that appear there, i.e. sum of monomials in numeral and denominator over all possible entries of the array
        n_mon = sum([length(i.den.exp)+length(i.num.exp) for i in trop])
        n_monomials[i] = n_mon
    end
    return n_monomials
end 

"""
Experiment 3: Compute the number of linear regions of a neural network as architecture varies
"""
function untrained_linear_region_computations(architectures)
    # this array will store the number of linear regions for each architecture
    n_regions = zeros(length(architectures))
    for i in Base.eachindex(architectures)
        # with a random neural network with a given architecture 
        weights, bias, thresholds = random_mlp(architectures[i], false)
        # compute the tropical Puiseux rational map
        trop = mlp_to_trop(weights, bias, thresholds)[1]
        # count the number of linear regions of the network
        n_reg = length(enum_linear_regions_rat(trop.num, trop.den))
        n_regions[i] = n_reg
    end
    return n_regions 
end 


function trained_linear_region_computation()
    return false
end 
