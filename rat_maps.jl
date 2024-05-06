using Pkg 
Pkg.add("Oscar")
#Pkg.add("StatsBase")
Pkg.add("Combinatorics")
using Oscar
#using StatsBase
using Combinatorics

struct TropicalPuiseuxPoly
    coeff::Dict
    exp::Vector
end 

struct TropicalPuiseuxRational
    num::TropicalPuiseuxPoly
    den::TropicalPuiseuxPoly
end 

# TODO: implement lexicographic ordering of terms

function TropicalPuiseuxPoly(coeff::Dict, exp::Vector, sorted)
    # first we need to order everything lexicographically
    if !sorted 
        exp = sort(exp)
    end 
    return TropicalPuiseuxPoly(coeff, exp)
end 

function TropicalPuiseuxPoly(coeff::Vector, exp::Vector, sorted=false)
    if !sorted 
        I = sortperm(exp)
        exp = exp[I]
        coeff = coeff[I]
    end 
    return TropicalPuiseuxPoly(Dict(zip(exp, coeff)), exp)
end 


function TropicalPuiseuxPoly_zero(n, f)
    exp = [Base.zeros(n)]
    coeff = Dict(Base.zeros(n) => zero(f.coeff[f.exp[1]]))
    return TropicalPuiseuxPoly(coeff, exp)
end 

function eachindex(f::TropicalPuiseuxPoly)
    return Base.eachindex(f.exp)
end 

function Oscar.nvars(f::TropicalPuiseuxPoly)
    if !is_empty(f.coeff)
        return length(f.exp[1])
    else 
        return -1
    end 
end 

function Base.:+(f::TropicalPuiseuxPoly, g::TropicalPuiseuxPoly)
    """
    Takes two TropicalPuiseuxPoly whose exponents are lexicographically ordered and outputs the sum with 
    lexicoraphically ordered exponents
    """
    lf = length(f.coeff)
    lg = length(g.coeff)
    # initialise coeffs and exponents vectors for the sum h = f + g
    h_coeff = Dict()
    h_exp = Vector()
    # initialise indexing variable for loops below
    j=1
    # at each term of g, check if there is a term of f with matching exponents
    for i in eachindex(g)
        # if we have exchausted all terms of f then we need to add all the remaining terms of g
        if j > lf 
               c = g.exp[i] 
               h_coeff[c] = g.coeff[c]
               push!(h_exp, c)
        else 
            # loop through terms of f ordered lexicographically, until we reach a term with a larger power 
            while j <= lf
                c = g.exp[i] 
                d = f.exp[j]
                if d > c 
                    # if c > d then we have reached the first exponent of f larger than c so we can stop here, and add the i-th 
                    # term of g to h.
                    h_coeff[c] = g.coeff[c]
                    push!(h_exp, c)
                    break
                elseif c == d 
                    # if we reach an equal exponent, both get added simultaneously to the sum.
                    h_coeff[c] = f.coeff[c]+g.coeff[c]
                    push!(h_exp, c)
                    # update j for the iteration with the next i
                    j+=1
                    break
                else 
                    # if d < c then we can add that exponent of f to the sum
                    h_coeff[d] = f.coeff[d]
                    push!(h_exp, d)
                    #if j < lf 
                    j+=1
                end 
                # Note about the indexing variable j:
                # Since a iteration i, we stop when we have either reached a j whose corresponding exponent is too large, or 
                # equal to that of i, we can start the iteration of i+1 at the j at which the previous iteration stopped.
            end 
        end
    end 
    # once we have exhausted all terms of g, we need to check for remaining terms of f
    while j <= lf 
        d = f.exp[j]
        h_coeff[d] = f.coeff[d]
        push!(h_exp, d)
        j+=1
    end 
    h = TropicalPuiseuxPoly(h_coeff, h_exp, true)
    return h
end 

# Lexicographic product of lexicographically ordered puisieux polynomials
# This is currently buggy and should be fixed.
function Base.:*(f::TropicalPuiseuxPoly, g::TropicalPuiseuxPoly)
    prod = TropicalPuiseuxPoly_zero(nvars(f), f)
    for i in eachindex(g)
        term_coeff = Dict()
        for (key, elem) in f.coeff
            term_coeff[key+g.exp[i]] = g.coeff[g.exp[i]] * elem
        end 
        term_exp = [g.exp[i] + f.exp[j] for j in eachindex(f)]
        prod += TropicalPuiseuxPoly(term_coeff, term_exp, true)
    end 
    return prod 
end 

function Base.string(f::TropicalPuiseuxPoly)
    str = ""
    if nvars(f)==1
        for i in eachindex(f)
            if i == 1 
                str *= repr(f.coeff[f.exp[i]]) * "*T^" * repr(f.exp[i][1])
            else 
                str *= " + " * repr(f.coeff[f.exp[i]]) * "*T^" * repr(f.exp[i][1])
            end 
        end 
    else 
        for i in eachindex(f)
            if i == 1
                str *= repr(f.coeff[f.exp[i]]) 
            else 
                str *= " + " * repr(f.coeff[f.exp[i]]) 
            end 
            exp = f.exp[i]
            for j in eachindex(exp)
                str *= "*T_" * repr(j) + "^" * repr(exp[j])   
            end
        end
    end 
    return str
end 

function Base.repr(f::TropicalPuiseuxPoly)
    return string(f)
end 

"""# implement me 
function Base.comp(f::TropicalPuissieuxRational, g::TropicalPuissieuxRational)
    return false 
end 

# implement me 
function Base.:+(f::TropicalPuissieuxRational, g::TropicalPuissieuxRational)
    return false
end 

# implement me 
function Base.:*(f::TropicalPuissieuxRational, g::TropicalPuissieuxRational)
    return false
end 

# implement me 
function Base.comp(f::TropicalPuissieuxRational, g::TropicalPuissieuxRational)
    return false 
end """


###### UNIT TESTS ###########

R = tropical_semiring(max)

f_coeff = [R(1), R(2)]
g_coeff = [R(2), R(1)]
h_coeff = [R(1), R(8)]
f_exp = [[1.0], [0.0]]
g_exp = [[1.0], [0.0]]
h_exp = [[1.2], [4.8]]

f = TropicalPuiseuxPoly(f_coeff, f_exp)
g = TropicalPuiseuxPoly(g_coeff, g_exp)
h = TropicalPuiseuxPoly(h_coeff, h_exp)

println(string(f*f))