using Pkg 
using Oscar
using Combinatorics
using Distributions
include("rat_maps.jl")

# takes in weight matrix A, bias term b and activation threshold t and outputs vector of tropical rational functions  
function single_to_trop(A, b, t)
    G = Vector{TropicalPuiseuxRational}()
    if size(A, 1) != length(b) || size(A, 1) != length(t) 
        println("Dimensions of matrix don't agree with constant term or threshold")
        return false
    end 
    R = tropical_semiring(max)
    # first make sure that the entries of b are elements of the tropical semiring
    b = [R(Rational(i)) for i in b]
    # and same for t
    t = [R(Rational(i)) for i in t]
    sizehint!(G, size(A, 1))
    for i in axes(A, 1)
        # first split the i-th line of A into its positive and negative components
        pos = zeros(size(A, 2))
        neg = zeros(size(A, 2))
        for j in axes(A, 2)
            pos[j] = max(A[i, j], 0)
            neg[j] = max(-A[i, j], 0)
        end
        # the numerator is the monomial given by the positive part, with coeff b[i], plus the monomial given by the negative part 
        # with coeff t[i]
        num = TropicalPuiseuxMonomial(b[i], pos) + TropicalPuiseuxMonomial(t[i], neg)
        # the denominator is the monomila given by the negative part, with coeff the tropical multiplicative 
        # unit, i.e. 0
        den = TropicalPuiseuxMonomial(one(t[i]), neg)
        push!(G, num/den) 
    end 
    return G
end     

"""
mlp_to_trop(linear_maps, bias, thresholds) computes the tropical Puiseux rational function associated to a multilayer perceptron.

inputs: linear maps: an array containing the weight matrices of the neural network. 
        bias: an array containing the biases at each layer
        thresholds: an array containing the threshold of the activation function at each layer, i.e. the number t such that the activation is of
        the form x => max(x,t).
outputs: an object of type TropicalPuiseuxRational.
"""
function mlp_to_trop(linear_maps, bias, thresholds)
    R = tropical_semiring(max)
    # initialisation: the first vector of tropical rational functions is just the identity function
    output = TropicalPuiseuxRational_identity(size(linear_maps[1], 2), R(1))
    # iterate through the layers and compose variable output with the current layer at each step
    for i in Base.eachindex(linear_maps)
        A = linear_maps[i]
        b = bias[i]
        t = thresholds[i]
        #check sizes agree
        if size(A, 1) != length(b) || size(A, 1) != length(t) 
            # stricly speaking this should be implemented as an exception
            println("Dimensions of matrix don't agree with constant term or threshold")
        end 
        # compute the vector of tropical rational functions corresponding to the function 
        # max(Ax+b, t) where A = linear_maps[i], b = bias[i] and t = thresholds[i]
        ith_tropical = single_to_trop(A, b, t)
        # compose this with the output of the previous layer
        output = comp(ith_tropical, output)
    end 
    return output
end 

"""
random_mlp(dims, random_thresholds) returns a multilayer perceptron with architecture specified by the array dims and random weights.

inputs: dims: array of integers specifying the width of each layer
        random_thresholds: boolean. If set to true, the threshold of the activation function at each layer is chosen at random. Otherwise 
            the thresholds are all set to 0, i.e. all the activation functions are the ReLU function. Default value is false.
"""
function random_mlp(dims, random_thresholds=false)
    # Use He initialisation, i.e. we sample weights with distribution N(0, sqrt(2/n))
    weights = [rand(Normal(0, sqrt(2/dims[1])), dims[i+1], dims[i]) for i in 1:length(dims)-1]
    biases = [rand(Normal(0, sqrt(2/dims[1])), dims[i]) for i in 2:length(dims)]
    if random_thresholds
        thresholds = [rand(dims[i]) for i in 2:length(dims)]
    else 
        thresholds = [zeros(dims[i]) for i in 2:length(dims)]
    end 
    return (weights, biases, thresholds)
end 

######### Unit tests #############
function test_mlp_to_trop()
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

    weights, bias, thresholds = random_mlp([5, 5, 2], true)
    trop = mlp_to_trop(weights, bias, thresholds)
    println("done")
    #println(trop)
    #println(weights)
    #@show length(trop)
    R = tropical_semiring(max)

    A1 = [3.0 -5.0;
            7.0 8.0]

    b1 = [R(3),
        R(7)]

    t1 = [ R(0),
        R(0)]

    g = single_to_trop(A1, b1, t1)

    for i in g
        println(string(i))
    end 
end 