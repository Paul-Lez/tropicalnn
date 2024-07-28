include("rat_maps.jl")
include("linear_regions.jl")

function monomial_elim(f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    new_exp = Vector{Vector{T}}()
    sizehint!(new_exp, length(f.exp))
    new_coeff = Dict()
    for i in Base.eachindex(f.exp)
        if Oscar.is_feasible(polyhedron(f, i))
            e = f.exp[i] 
            push!(new_exp, e)
            new_coeff[e] = f.coeff[e]
        #else 
         #   println("Caught useless monomial")
        end 
    end 
    return TropicalPuiseuxPoly(new_coeff, new_exp)
end 

function monomial_elim(f::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return TropicalPuiseuxRational(monomial_elim(f.num), monomial_elim(f.den))
end

function monomial_elim(F::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return [monomial_elim(f) for f in F]
end

function mlp_to_trop_with_elim(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
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
            output = monomial_elim(output)
            #output = dedup_monomials(output)
        end 
    end 
    return output
end 


function mlp_to_trop_with_quasi_elim(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
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
            if i != length(linear_maps)
                output = monomial_elim(output)
                println("here ")
            end 
            #output = dedup_monomials(output)
        end 
    end 
    return output
end 

function mlp_to_trop_with_double_elim(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
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
            ith_tropical = monomial_elim(single_to_trop(A, b, t))
            # compose this with the output of the previous layer
            output = comp(ith_tropical, output)
           # output = monomial_elim(output)
            println("here ")
            #output = dedup_monomials(output)
        end 
    end 
    return output
end 