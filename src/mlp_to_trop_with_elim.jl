include("rat_maps.jl")
include("linear_regions.jl")
using Oscar

# removes redundant monomials from a tropical polynomial 
function monomial_elim(f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    new_exp = Vector{Vector{T}}()
    sizehint!(new_exp, length(f.exp))
    new_coeff = Dict()
    for i in Base.eachindex(f.exp)
        if Oscar.is_feasible(polyhedron(f, i))
            e = f.exp[i] 
            push!(new_exp, e)
            new_coeff[e] = f.coeff[e]
        end 
    end 
    return TropicalPuiseuxPoly(new_coeff, new_exp)
end 

# removes redundant monomials from rational function
function monomial_elim(f::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return TropicalPuiseuxRational(monomial_elim(f.num), monomial_elim(f.den))
end

# removes redundant monomials from vector of rational functions
function monomial_elim(F::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return [monomial_elim(f) for f in F]
end

# removes redundant monomials from a tropical polynomial 
function monomial_strong_elim(f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    new_exp = Vector{Vector{T}}()
    sizehint!(new_exp, length(f.exp))
    new_coeff = Dict()
    for i in Base.eachindex(f.exp)
        poly = polyhedron(f, i)
        if Oscar.is_fulldimensional(poly) #Oscar.is_feasible(poly) && Oscar.dim(poly) == nvars(f)
            e = f.exp[i] 
            push!(new_exp, e)
            new_coeff[e] = f.coeff[e]
        #else 
         #   println("Caught useless monomial")
        end 
    end 
    return TropicalPuiseuxPoly(new_coeff, new_exp)
end 

# removes redundant monomials from a rational function
function monomial_strong_elim(f::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return TropicalPuiseuxRational(monomial_strong_elim(f.num), monomial_strong_elim(f.den))
end

# removes redundant monomials from a vector of rational functions
function monomial_strong_elim(F::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return [monomial_strong_elim(f) for f in F]
end

function mlp_to_trop_with_elim(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    """
    mlp_to_trop_with_elim(linear_maps, bias, thresholds) computes the tropical Puiseux rational function associated to a multilayer perceptron, and runs monomial_elim at each layer.
    
    inputs: linear maps: an array containing the weight matrices of the neural network. 
            bias: an array containing the biases at each layer
            thresholds: an array containing the threshold of the activation function at each layer, i.e. the number t such that the activation is of
            the form x => max(x,t).
    outputs: an object of type Vector{TropicalPuiseuxRational}
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
    mlp_to_trop_quasi_elim(linear_maps, bias, thresholds) computes the tropical Puiseux rational function associated to a multilayer perceptron, and runs monomial_elim at each layer except the last.
    
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
        println("Computation is at layer ", i)
        #check sizes agree
        if size(A, 1) != length(b) || size(A, 1) != length(t) 
            # stricly speaking this should be implemented as an exception
            println("Dimensions of matrix don't agree with constant term or threshold")
        end 
        if i != 1 
            # compute the vector of tropical rational functions corresponding to the function 
            # max(Ax+b, t) where A = linear_maps[i], b = bias[i] and t = thresholds[i]
            ith_tropical = single_to_trop(A, b, t)
            monomial_elim(ith_tropical)
            # compose this with the output of the previous layer
            output = comp(ith_tropical, output)
            if i != length(linear_maps)
                output = monomial_elim(output)
            end 
            println("Number of monomials at layer ", i, " is ", monomial_count(output))
            #output = dedup_monomials(output)
        end 
    end 
    return output
end 

# experimental, do not use.
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

function mlp_to_trop_with_strong_elim(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
"""
mlp_to_trop_with_strong_elim(linear_maps, bias, thresholds) computes the tropical Puiseux rational function associated to a multilayer perceptron, and runs monomial_strong_elim at each layer to remove redundant monomials.

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
        output = monomial_strong_elim(output)
        #output = dedup_monomials(output)
    end 
end 
return output
end 

function mlp_to_trop_with_quicksum_with_strong_elim(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    """
    mlp_to_trop_with_quicksum_with_strong_elim(linear_maps, bias, thresholds) computes the tropical Puiseux rational function associated to a multilayer perceptron. Runs monomial_strong_elim at each layer, and uses quicksum operations for tropical objects.
    
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
            println("  Currently working on layer ", i)
            #check sizes agree
            if size(A, 1) != length(b) || size(A, 1) != length(t) 
                # stricly speaking this should be implemented as an exception
                println("Dimensions of matrix don't agree with constant term or threshold")
            end 
            if i != 1
                # compute the vector of tropical rational functions corresponding to the function 
                # max(Ax+b, t) where A = linear_maps[i], b = bias[i] and t = thresholds[i]
                println("   Tropicalising the linear map")
                ith_tropical = single_to_trop(A, b, t)
                # compose this with the output of the previous layer
                println("   Calculating the composition")
                output = comp_with_quicksum(ith_tropical, output)
                println("   Non-reduced form has ", monomial_count(output), " monomials")
                println("   Eliminating useless monomials")
                output = monomial_strong_elim(output)
                println("   Monomial count at layer ", i, " is ", monomial_count(output))
            end 
        end 
    return output
end 

function mlp_to_trop_with_quicksum(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    """
    mlp_to_trop_with_quicksum(linear_maps, bias, thresholds) computes the tropical Puiseux rational function associated to a multilayer perceptron. Uses quicksum operations for tropical objects.
    
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
            println("  Currently working on layer ", i)
            #check sizes agree
            if size(A, 1) != length(b) || size(A, 1) != length(t) 
                # stricly speaking this should be implemented as an exception
                println("Dimensions of matrix don't agree with constant term or threshold")
            end 
            if i != 1
                # compute the vector of tropical rational functions corresponding to the function 
                # max(Ax+b, t) where A = linear_maps[i], b = bias[i] and t = thresholds[i]
                println("   Tropicalising the linear map")
                ith_tropical = single_to_trop(A, b, t)
                # compose this with the output of the previous layer
                println("   Calculating the composition")
                output = comp_with_quicksum(ith_tropical, output)
                println("   Monomial count at layer ", i, " is ", monomial_count(output))
            end 
        end 
    return output
end 