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

function TropicalPuiseuxPoly_const(n, c)
    exp = [Base.zeros(n)]
    coeff = Dict(Base.zeros(n) => c)
    return TropicalPuiseuxPoly(coeff, exp)
end 


function TropicalPuiseuxPoly_zero(n, f)
    return TropicalPuiseuxPoly_const(n, zero(f.coeff[f.exp[1]]))
end 

function TropicalPuiseuxPoly_one(n, f::TropicalPuiseuxPoly)
    return TropicalPuiseuxPoly_one(n, zero(f.coeff[f.exp[1]]))
end 

function TropicalPuiseuxPoly_one(n, c::TropicalSemiringElem)
    return TropicalPuiseuxPoly_const(n, zero(c))
end 


function TropicalPuiseuxMonomial(c, exp)
    return TropicalPuiseuxPoly([c for _ in 1:length(exp)], [exp], true)
end 

function TropicalPuiseuxPoly_to_rational(f)
    return TropicalPuiseuxRational(f, TropicalPuiseuxPoly_one(nvars(f), f))
end 

function TropicalPuiseuxPoly_zero(n, f)
    exp = [Base.zeros(n)]
    coeff = Dict(Base.zeros(n) => zero(f.coeff[f.exp[1]]))
    return TropicalPuiseuxPoly(coeff, exp)
end 

function TropicalPuiseuxRational_identity(n, c)
    output = Vector{TropicalPuiseuxRational}()
    sizehint!(output, n)
    for i in 1:n 
        # add the i-th coordinate viewed as a tropical rational function 
        push!(output, TropicalPuiseuxPoly_to_rational( 
            TropicalPuiseuxMonomial(one(c), [j == i ? 1 : 0 for j in 1:n])))
    end 
    return output
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

function Oscar.nvars(f::TropicalPuiseuxRational)
    return Oscar.nvars(f.den)
end 

function TropicalPuiseuxRational_zero(n, f)
    return TropicalPuiseuxRational(TropicalPuiseuxPoly_zero(n, f), TropicalPuiseuxPoly_one(n, f))
end 

function TropicalPuiseuxRational_one(n, f)
    return TropicalPuiseuxRational(TropicalPuiseuxPoly_one(n, f), TropicalPuiseuxPoly_one(n, f))
end 

# todo. how do we want to keep track of the sorting?
function Base.:/(f::TropicalPuiseuxPoly, g::TropicalPuiseuxPoly)
    return TropicalPuiseuxRational(f, g)
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
    for i in eachindex(f)
        # in dimension 1 we omit subscripts on the variables
        if nvars(f)==1
            if i == 1 
                str *= repr(f.coeff[f.exp[i]]) * "*T^" * repr(f.exp[i][1])
            else 
                str *= " + " * repr(f.coeff[f.exp[i]]) * "*T^" * repr(f.exp[i][1])
            end 
        else 
            if i == 1
                str *= repr(f.coeff[f.exp[i]]) 
            else 
                str *= " + " * repr(f.coeff[f.exp[i]]) 
            end 
            exp = f.exp[i]
            for j in Base.eachindex(exp)
                str *= "*T_" * repr(j) * "^" * repr(exp[j])   
            end 
        end
    end
    return str
end 

function Base.repr(f::TropicalPuiseuxPoly)
    return string(f)
end 

function Base.string(f::TropicalPuiseuxRational)
    return string(f.num) * " / " * string(f.den)
end 

# exponentiation of a tropical Puiseux polynomial by a positive rational
function Base.:^(f::TropicalPuiseuxPoly, rat::Float64)
    new_f_coeff = Dict()
    new_f_exp = copy(f.exp)
    new_f_exp = rat * new_f_exp 
    for (key, elem) in f.coeff
        new_f_coeff[rat*key] = elem
    end 
    return TropicalPuiseuxPoly(new_f_coeff, new_f_exp, true)
end 

# exponentiation of a tropical Puiseux rational function by a positive rational
function Base.:^(f::TropicalPuiseuxRational, rat::Float64)
    return TropicalPuiseuxRational(f.num^rat , f.den^rat)
end 

function Base.:^(f::TropicalPuiseuxPoly, int::Int64)
    new_f_coeff = Dict()
    new_f_exp = copy(f.exp)
    new_f_exp = int * new_f_exp 
    for (key, elem) in f.coeff
        new_f_coeff[int*key] = elem
    end 
    return TropicalPuiseuxPoly(new_f_coeff, new_f_exp, true)
end 

# exponentiation of a tropical Puiseux rational function by a positive integer
function Base.:^(f::TropicalPuiseuxRational, int::Int64)
    return TropicalPuiseuxRational(f.num^int , f.denom^int)
end 

function Base.:*(a::TropicalSemiringElem, f::TropicalPuiseuxPoly)
    new_f_coeff = copy(f.coeff)
    new_f_exp = copy(f.exp)
    for i in eachindex(f)
        new_f_coeff[f.exp[i]] = a*f.coeff[f.exp[i]]
    end 
    return TropicalPuiseuxPoly(new_f_coeff, new_f_exp, true)
end 

function Base.:*(a::TropicalSemiringElem, f::TropicalPuiseuxRational)
    return TropicalPuiseuxRational(a*f.num, f.den)
end 

# univariate case 
function eval(f::TropicalPuiseuxPoly, a)
    zero(a)
    for (key, val) in f.coeff
        comp += val * a^key[1]
    end 
    return comp
end 

# evaluation in the multivariate case
function eval(f::TropicalPuiseuxPoly, G::Vector)
    comp = zero(G[1])
    for (key, val) in f.coeff
        term = one(a)
        for i in eachindex(G)
            term *= G[i]^key[i]    
        end 
        comp += val * term 
    end 
    return comp
end 

# Here f needs to be univariate 
function comp(f::TropicalPuiseuxPoly, g::TropicalPuiseuxPoly)
    comp = TropicalPuiseuxPoly_zero(nvars(g), f)
    for (key, val) in f.coeff
        comp += val * g^key[1]
    end 
    return comp
end 

# composition in the multivariate case
function comp(f::TropicalPuiseuxPoly, G::Vector{TropicalPuiseuxPoly})
    comp = TropicalPuiseuxPoly_zero(nvars(G[0]), f)
    for (key, val) in f.coeff
        term = TropicalPuiseuxPoly_one(nvars(G[0]), f)
        for i in eachindex(G)
            term *= G[i]^key[i]    
        end 
        comp += val * term 
    end 
    return comp
end 

function Base.:+(f::TropicalPuiseuxRational, g::TropicalPuiseuxRational)
    num = f.num * g.den + f.den * g.num 
    den = f.den * g.den 
    return TropicalPuiseuxRational(num, den) 
end 

# implement me 
function Base.:*(f::TropicalPuiseuxRational, g::TropicalPuiseuxRational)
    num = f.num * g.num 
    den = f.den * g.den 
    return TropicalPuiseuxRational(num, den)
end 

function Base.:/(f::TropicalPuiseuxRational, g::TropicalPuiseuxRational)
    num = f.num*g.den 
    den = f.den*g.den
    return TropicalPuiseuxRational(num, den)
end 


#univariate case
function comp(f::TropicalPuiseuxPoly, g::TropicalPuiseuxRational)
    comp = TropicalPuiseuxRational_zero(nvars(g), f)
    for (key, val) in f.coeff
        comp += val * g^key[1]
    end 
    return comp
end 

# multivariate case
function comp(f::TropicalPuiseuxPoly, G::Vector{TropicalPuiseuxRational})   
    comp = TropicalPuiseuxRational_zero(nvars(G[1]), f)
    for (key, val) in f.coeff
        term = TropicalPuiseuxRational_one(nvars(G[1]), f)
        for i in Base.eachindex(G)
            term *= G[i]^key[i]    
        end 
        comp += val * term 
    end 
    return comp
end 

# implement me 
function comp(f::TropicalPuiseuxRational, g::TropicalPuiseuxRational)
    return comp(f.num, g) / comp(f.den, g)
end

# multivariate case 
function comp(f::TropicalPuiseuxRational, G::Vector{TropicalPuiseuxRational})
    return comp(f.num, G) / comp(f.den, G)
end 

# double multivariate case 
function comp(F::Vector{TropicalPuiseuxRational}, G::Vector{TropicalPuiseuxRational})
    return [comp(f, G) for f in F]
end 

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

println(string(f^4))
println(string(comp(f, g)))