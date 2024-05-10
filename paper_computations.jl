using Pkg 
using Oscar
using Combinatorics
include("rat_maps.jl")
include("linear_regions.jl")
include("mlp_to_trop.jl")

n_terms = 50
n_variables = 5

for i in 1:2
    c_f = rand(Float64, n_terms)
    c_g = rand(Float64, n_terms)
    exp_f = []
    exp_g = []
    for j in 1:n_terms 
        push!(exp_f, rand(Float64, n_variables))
        push!(exp_g, rand(Float64, n_variables))
    end
    #println(exp_f)
    f = TropicalPuiseuxPoly(c_f, exp_f)
    g = TropicalPuiseuxPoly(c_g, exp_g)
    lins = enum_linear_regions_rat(f, g)
    println(length(lins))
end 