using Pkg 
using Oscar
using Combinatorics
using Distributions
include("rat_maps.jl")

function single_to_trop(A::Matrix{T}, b, t) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
"""
Inputs: weight matrix A, bias term b and activation threshold t
Outputs: vector of tropical Puiseux rational functions that express the function max(Ax+b, t) as a tropical Puiseux rational function.
""" 
    G = Vector{TropicalPuiseuxRational{T}}()
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
        pos = Vector{T}()
        neg = Vector{T}()
        for j in axes(A, 2)
            push!(pos, max(A[i, j], 0))
            push!(neg, max(-A[i, j], 0))
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

function mlp_to_trop(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
"""
mlp_to_trop(linear_maps, bias, thresholds) computes the tropical Puiseux rational function associated to a multilayer perceptron.

inputs: linear maps: an array containing the weight matrices of the neural network. 
        bias: an array containing the biases at each layer
        thresholds: an array containing the threshold of the activation function at each layer, i.e. the number t such that the activation is of
        the form x => max(x,t).
outputs: an object of type TropicalPuiseuxRational.
"""
    R = tropical_semiring(max)
    # initialisation: the first vector of tropical rational functions is just the identity function
    output = single_to_trop(linear_maps[1], bias[1], thresholds[1])
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
        if i != 1
            # compute the vector of tropical rational functions corresponding to the function 
            # max(Ax+b, t) where A = linear_maps[i], b = bias[i] and t = thresholds[i]
            ith_tropical = single_to_trop(A, b, t)
            # compose this with the output of the previous layer
            output = comp(ith_tropical, output)
        end 
    end 
    return output
end 

function mlp_to_trop_with_quicksum(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    """
    mlp_to_trop(linear_maps, bias, thresholds) computes the tropical Puiseux rational function associated to a multilayer perceptron.
    
    inputs: linear maps: an array containing the weight matrices of the neural network. 
            bias: an array containing the biases at each layer
            thresholds: an array containing the threshold of the activation function at each layer, i.e. the number t such that the activation is of
            the form x => max(x,t).
    outputs: an object of type TropicalPuiseuxRational.
    """
        R = tropical_semiring(max)
        # initialisation: the first vector of tropical rational functions is just the identity function
        output = single_to_trop(linear_maps[1], bias[1], thresholds[1])
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
            if i != 1
                # compute the vector of tropical rational functions corresponding to the function 
                # max(Ax+b, t) where A = linear_maps[i], b = bias[i] and t = thresholds[i]
                ith_tropical = single_to_trop(A, b, t)
                # compose this with the output of the previous layer
                output = comp_with_quicksum(ith_tropical, output)
            end 
        end 
    return output
end 

function mlp_to_trop_with_mul_with_quicksum(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    """
    mlp_to_trop(linear_maps, bias, thresholds) computes the tropical Puiseux rational function associated to a multilayer perceptron.
    
    inputs: linear maps: an array containing the weight matrices of the neural network. 
            bias: an array containing the biases at each layer
            thresholds: an array containing the threshold of the activation function at each layer, i.e. the number t such that the activation is of
            the form x => max(x,t).
    outputs: an object of type TropicalPuiseuxRational.
    """
        R = tropical_semiring(max)
        # initialisation: the first vector of tropical rational functions is just the identity function
        output = single_to_trop(linear_maps[1], bias[1], thresholds[1])
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
            if i != 1
                # compute the vector of tropical rational functions corresponding to the function 
                # max(Ax+b, t) where A = linear_maps[i], b = bias[i] and t = thresholds[i]
                ith_tropical = single_to_trop(A, b, t)
                # compose this with the output of the previous layer
                output = comp_with_quicksum(ith_tropical, output)
            end 
        end 
    return output
end 

function random_mlp(dims, random_thresholds=false, symbolic=true)
    """
    random_mlp(dims, random_thresholds) returns a multilayer perceptron with architecture specified by the array dims and random weights.
    
    inputs: dims: array of integers specifying the width of each layer
            random_thresholds: boolean. If set to true, the threshold of the activation function at each layer is chosen at random. Otherwise 
                the thresholds are all set to 0, i.e. all the activation functions are the ReLU function. Default value is false.
    """
    # if symbolic is set to true then we work with symbolic fractions. 
    if symbolic 
        # Use He initialisation, i.e. we sample weights with distribution N(0, sqrt(2/n))
        weights = [Rational{BigInt}.(rand(Normal(0, sqrt(2/dims[1])), dims[i+1], dims[i])) for i in 1:length(dims)-1]
        biases = [Rational{BigInt}.(rand(Normal(0, sqrt(2/dims[1])), dims[i])) for i in 2:length(dims)]
        if random_thresholds
            thresholds = [Rational{BigInt}.(rand(dims[i])) for i in 2:length(dims)]
        else 
            thresholds = [Rational{BigInt}.(zeros(dims[i])) for i in 2:length(dims)]
        end 
    else # otherwise we work with Floats
        # Use He initialisation, i.e. we sample weights with distribution N(0, sqrt(2/n))
        weights = [rand(Normal(0, sqrt(2/dims[1])), dims[i+1], dims[i]) for i in 1:length(dims)-1]
        biases = [rand(Normal(0, sqrt(2/dims[1])), dims[i]) for i in 2:length(dims)]
        if random_thresholds
            thresholds = [rand(dims[i]) for i in 2:length(dims)]
        else 
            thresholds = [zeros(dims[i]) for i in 2:length(dims)]
        end 
    end 
    return (weights, biases, thresholds)
end 

function mlp_to_trop_with_dedup(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    """
    mlp_to_trop(linear_maps, bias, thresholds) computes the tropical Puiseux rational function associated to a multilayer perceptron.
    
    inputs: linear maps: an array containing the weight matrices of the neural network. 
            bias: an array containing the biases at each layer
            thresholds: an array containing the threshold of the activation function at each layer, i.e. the number t such that the activation is of
            the form x => max(x,t).
    outputs: an object of type TropicalPuiseuxRational.
    """
        R = tropical_semiring(max)
        # initialisation: the first vector of tropical rational functions is just the identity function
        output = single_to_trop(linear_maps[1], bias[1], thresholds[1])
        output = dedup_monomials(output)
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
            if i != 1
                # compute the vector of tropical rational functions corresponding to the function 
                # max(Ax+b, t) where A = linear_maps[i], b = bias[i] and t = thresholds[i]
                ith_tropical = single_to_trop(A, b, t)
                # compose this with the output of the previous layer
                output = comp(ith_tropical, output)
                output = dedup_monomials(output)
            end 
        end 
        return output
    end 