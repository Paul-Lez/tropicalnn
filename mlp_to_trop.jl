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
    for j in axes(A, 2)
        # first split the i-th line of A into its positive and negative components
        pos = zeros(size(A, 1))
        neg = zeros(size(A, 1))
        for i in axes(A, 1)
            pos[i] = max(A[i, j], 0)
            neg[i] = max(-A[i, j], 0)
        end
        #println(A[:, j])
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
    for i in Base.eachindex(linear_maps)
        # compute the vector of tropical rational functions corresponding to the function 
        # max(Ax+b, t) where A = linear_maps[i], b = bias[i] and t = thresholds[i]
        ith_tropical = single_to_trop(linear_maps[i], bias[i], thresholds[i])
        # compose this with the output of the previous layer
        println("Doing composition ", i)
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
#p = mlp_to_trop([A1, A2], [b1, b2], [t1, t2])[1]
#println(string(p))

d1 = 5
d2 = 2
d3 = 2
d4 = 10
d5 = 1

A1 = rand(d1, d2)
A2 = rand(d2, d3)
A3 = rand(d3, d4)
A4 = rand(d4, d5)

b1 = [R(QQ(Rational(i))) for i in rand(Float32, d2)]
b2 = [R(QQ(Rational(i))) for i in rand(Float32, d3)]
b3 = [R(QQ(Rational(i))) for i in rand(Float32, d4)]
b4 = [R(QQ(Rational(i))) for i in rand(Float32, d5)]

t1 = [R(QQ(Rational(i))) for i in rand(Float32,d2)]
t2 = [R(QQ(Rational(i))) for i in rand(Float32, d3)]
t3 = [R(QQ(Rational(i))) for i in rand(Float32, d4)]
t4 = [R(QQ(Rational(i))) for i in rand(Float32, d5)]

p = mlp_to_trop([A1, A2, A3, A4], [b1, b2, b3, b4], [t1, t2, t3, t4])[1] #+ mlp_to_trop([A1, A2, A3, A4], [b1, b2, b3, b4], [t1, t2, t3, t4])[2]
arr1 = copy(p.num.exp)
arr2 = copy(p.den.exp)
println(length(arr1))
println(length(arr2))
filter!(e -> p.num.coeff[e] != zero(R), arr1)
filter!(e -> p.den.coeff[e] != zero(R), arr2)
println(length(arr1))
println(length(arr2))
