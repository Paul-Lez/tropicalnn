using Pkg 
Pkg.add("Oscar")
#Pkg.add("StatsBase")
Pkg.add("Combinatorics")
using Oscar
#using StatsBase
using Combinatorics
include("rat_maps.jl")
#using rat_maps

# takes in weight matrix A, bias term b and activation threshold t and outputs vector of tropical rational functions  
function single_to_trop(A, b, t)
    G = Vector{TropicalPuiseuxRational}()
    sizehint!(G, size(A, 2))
    for j in stride(A, 2)
        # first split the i-th line of A into its positive and negative components
        pos = zeros(size(A, 1))
        neg = zeros(size(A, 1))
        for i in stride(A, 1)
            pos[i] = max(A[i][j], 0)
            neg[i] = max(-A[i][j], 0)
        end
        # the numerator is the monomial given by the positive part, with coeff b[i], plus the monomial given by the negative part 
        # with coeff t[i]
        num = TropicalPuiseuxMonomial(b[j], pos) + TropicalPuiseuxMonomial(t[j], neg)
        # the denominator is the monomila given by the negative part, with coeff the tropical multiplicative 
        # unit, i.e. 0
        den = TropicalPuiseuxMonomial(one(t[j]), neg)
        push!(G, num/den) 
    end 
    return G
end     

function mlp_to_trop(linear_maps, bias, thresholds)
    # initialisation: the first vector of tropical rational functions is just the identity function
    output = TropicalPuiseuxRational_identity(size(linear_maps[1], 1), bias[1][1])
    # iterate through the layers and compose variable output with the current layer at each step
    for i in stride(linear_maps, 1)
        # compute the vector of tropical rational functions corresponding to the function 
        # max(Ax+b, t) where A = linear_maps[i], b = bias[i] and t = thresholds[i]
        ith_tropical = single_to_trop(linear_maps[i], bias[i], thresholds[i])
        # compose this with the output of the previous layer
        output = comp(ith_tropical, output)
    end 
    return output
end 



######### Unit tests #############
R = tropical_semiring(max)
A1 = [[1.0, 1.0],
     [2.0, 2.2 ]]

A2 = [[1], [2]]

b1 = [R(1),
     R(2)]

b2 = [R(1)]

t1 = [R(2), 
      R(2)]

t2 = [R(0)]

#println(single_to_trop(A1, b1, t1))
p = mlp_to_trop([A1, A2], [b1, b2], [t1, t2])[1]
println(string(p))