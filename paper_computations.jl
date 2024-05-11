using Pkg 
using Oscar
using Combinatorics
using Plots
using JLD2
include("rat_maps.jl")
include("linear_regions.jl")
include("mlp_to_trop.jl")

# This file computes the (averaged over a few samples) number of linear regions of a tropical rational 
# function, for vatious numbers of terms and variables 

function computations(n_variables, n_terms, n_samples)
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

    JLD2.save_object("output_array.jld2", output)
end 

#plot(n_terms, output)

