using Pkg 
using Oscar
using Combinatorics
using RecursiveArrayTools


####################### BASIC CONSTRUCTIONS ##################################

struct TropicalPuiseuxPoly{T}
    coeff::Dict
    exp::Vector{Vector{T}}
end 

struct TropicalPuiseuxRational{T}
    num::TropicalPuiseuxPoly{T}
    den::TropicalPuiseuxPoly{T}
end 

# TODO: implement lexicographic ordering of terms

function TropicalPuiseuxPoly(coeff::Dict, exp::Vector{Vector{T}}, sorted) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
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

function TropicalPuiseuxPoly_const(n, c, f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    exp = [Base.zeros(T, n)]
    coeff = Dict(Base.zeros(T, n) => c)
    return TropicalPuiseuxPoly(coeff, exp)
end 


function TropicalPuiseuxPoly_zero(n::Int64, f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return TropicalPuiseuxPoly_const(n, zero(f.coeff[f.exp[1]]), f)
end 

function TropicalPuiseuxPoly_one(n::Int64, f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return TropicalPuiseuxPoly_one(n, one(f.coeff[f.exp[1]]), f)
end 

function TropicalPuiseuxPoly_one(n::Int64, c::TropicalSemiringElem, f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return TropicalPuiseuxPoly_const(n, one(c), f)
end 


function TropicalPuiseuxMonomial(c, exp::Vector{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return TropicalPuiseuxPoly([c for _ in 1:length(exp)], [exp], true)
end 

function TropicalPuiseuxPoly_to_rational(f)
    return TropicalPuiseuxRational(f, TropicalPuiseuxPoly_one(nvars(f), f))
end 

function TropicalPuiseuxPoly_zero(n, f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    exp = [Base.zeros(T, n)]
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

function TropicalPuiseuxRational_zero(n::Int64, f::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return TropicalPuiseuxRational(TropicalPuiseuxPoly_zero(n, f.num), TropicalPuiseuxPoly_one(n, f.den))
end 

function TropicalPuiseuxRational_one(n, f)
    return TropicalPuiseuxRational(TropicalPuiseuxPoly_one(n, f.num), TropicalPuiseuxPoly_one(n, f.num))
end 

##################################################################################

####################### STRING REPRESENTATIONS ###################################

function Base.string(f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
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
                str *= " * T_" * repr(j) * " ^ " * repr(exp[j])   
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

######################################################################

########################## EVALUATION ################################

#### This section defines API to evaluate Tropical Puiseux Polynomials ####

# univariate case 
function eval(f::TropicalPuiseuxPoly, a)
    comp = zero(a)
    for (key, val) in f.coeff
        comp += val * a^key[1]
    end 
    return comp
end 

# univariate case
function eval(f::TropicalPuiseuxRational, a)
    return eval(f.num, a) / eval(f.den, a)
end  

# multivariate case
function eval(f::TropicalPuiseuxPoly{T}, a::Vector) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    #R = tropical_semiring(max)
    ev = zero(a[1])
    for (exp, coeff) in f.coeff
        term = one(a[1])
        for i in Base.eachindex(a)
            term *= a[i]^exp[i]
        end
        ev += coeff * term
    end 
    return ev
end  

# multivariate case
function eval(f::TropicalPuiseuxRational{T}, a::Vector) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    n::TropicalSemiringElem{typeof(max)} = eval(f.num, a) 
    m::TropicalSemiringElem{typeof(max)} = eval(f.den, a) 
    return n / m
end 

function eval(F::Vector{TropicalPuiseuxRational{T}}, a::Vector) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return [eval(f, a) for f in F]
end 

# Test version of eval, do not use
function eval_temporary(f::TropicalPuiseuxPoly{T}, G::Vector{Rational{T}}) where T<:Integer
    comp = -Rational{BigInt}(Base.Inf)
    for (key, val) in f.coeff
        term = 0
        for i in Base.eachindex(G)
            term += G[i]*key[i]    
        end 
        comp = max(Rational{BigInt}(QQ(val)) + term, comp)
    end 
    return comp
end

# Test version of eval, do not use
function eval_temporary(f::TropicalPuiseuxPoly, G::Vector{TropicalSemiringElem{typeof(max)}})
    G_new = [Float64(Rational(G[i])) for i in Base.eachindex(G)]
    return eval_temporary(f, G_new)
end

# Test version of eval, do not use
function eval_temporary(f::TropicalPuiseuxRational, G::Vector)
    return eval_temporary(f.num, G) - eval_temporary(f.den, G)
end 

######################################################################

################ ARITHMETIC OPERATIONS ###############################

#### This section implements standard arithmetic operations for tropical ####
#### polynomials and rational functions                                  ####

function Base.:/(f::TropicalPuiseuxPoly{T}, g::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return TropicalPuiseuxRational(f, g)
end 

# Quicker version of addition for vectors of tropical polynomials 
function quicksum(F::Vector{TropicalPuiseuxPoly{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    R = tropical_semiring(max)
    terms = reduce(vcat, [f.exp for f in F])
    h_exp::Vector{Vector{T}} = terms
    #terms = unique(flatview(VectorOfArray([f.exp for f in F])))
    #sort!(terms)
    h_coeff = Dict()
    for exp in terms
        h_coeff[exp] = sum([f.coeff[exp] for f in F if haskey(f.coeff, exp)])
    end 
    # this is set to true because we don't need sorting with this algo?
    return TropicalPuiseuxPoly(h_coeff, h_exp, true)
end 

function Base.:+(f::TropicalPuiseuxPoly{T}, g::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    """
    Takes two TropicalPuiseuxPoly whose exponents are lexicographically ordered and outputs the sum with 
    lexicoraphically ordered exponents
    """
    lf = length(f.coeff)
    lg = length(g.coeff)
    # initialise coeffs and exponents vectors for the sum h = f + g
    h_coeff = Dict()
    h_exp = Vector{Vector{T}}()
    # initialise indexing variable for loops below
    j=1
    # at each term of g, check if there is a term of f with matching exponents
    for i in eachindex(g)
        c = g.exp[i] 
        added = false
        # loop through terms of f ordered lexicographically, until we reach a term with a larger power 
        while j <= lf
            d = f.exp[j]
            if d > c 
                if g.coeff[c] != zero(g.coeff[c])
                    # if c > d then we have reached the first exponent of f larger than c so we can stop here, and add the i-th 
                    # term of g to h.
                    h_coeff[c] = g.coeff[c]
                    push!(h_exp, c)
                    added = true
                end 
                break
            elseif c == d 
                if g.coeff[c] != zero(g.coeff[c]) || f.coeff[c] != zero(g.coeff[c])
                    # if we reach an equal exponent, both get added simultaneously to the sum.
                    h_coeff[c] = f.coeff[c]+g.coeff[c]
                    push!(h_exp, c)
                    added = true
                end 
                # update j for the iteration with the next i
                j+=1
                break
            else 
                if f.coeff[d] != zero(f.coeff[d])
                    # if d < c then we can add that exponent of f to the sum
                    h_coeff[d] = f.coeff[d]
                    push!(h_exp, d)
                end 
                j+=1
            end 
            # Note about the indexing variable j:
            # Since a iteration i, we stop when we have either reached a j whose corresponding exponent is too large, or 
            # equal to that of i, we can start the iteration of i+1 at the j at which the previous iteration stopped.
        end 
        # if we have exchausted all terms of f then we need to add all the remaining terms of g
        if !added && j > lf && g.coeff[c] != zero(g.coeff[c])
            h_coeff[c] = g.coeff[c]
            push!(h_exp, c)
        end 
    end 
    # once we have exhausted all terms of g, we need to check for remaining terms of f
    while j <= lf 
        d = f.exp[j]
        if f.coeff[d] != zero(f.coeff[d])
            h_coeff[d] = f.coeff[d]
            push!(h_exp, d)
        end 
        j+=1
    end 
    h = TropicalPuiseuxPoly(h_coeff, h_exp, true)
    return h
end 

function Base.:*(f::TropicalPuiseuxPoly{T}, g::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    prod = TropicalPuiseuxPoly_zero(nvars(f), f)
    # if f = a_0 + ... + a_n T^n and g = b_0 + ... + b_n then the product is 
    # the sum of all the b_i T^i * f
    for i in eachindex(g)
        term_coeff = Dict()
        # compute the coefficients of b_i T^i * f
        for (key, elem) in f.coeff
            term_coeff[key+g.exp[i]] = g.coeff[g.exp[i]] * elem
        end 
        # compute the exponenets of b_i T^i * f
        term_exp = [g.exp[i] + f.exp[j] for j in eachindex(f)]
        prod += TropicalPuiseuxPoly(term_coeff, term_exp, true)
        #println(string(TropicalPuiseuxPoly(term_coeff, term_exp, true)))
    end 
    return prod 
end 

function mul_with_quicksum(f::TropicalPuiseuxPoly{T}, g::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    # if f = a_0 + ... + a_n T^n and g = b_0 + ... + b_n then the product is 
    # the sum of all the b_i T^i * f
    summands = Vector{TropicalPuiseuxPoly{T}}()
    sizehint!(summands, length(g.exp))
    for i in eachindex(g)
        term_coeff = Dict()
        # compute the coefficients of b_i T^i * f
        for (key, elem) in f.coeff
            #println(nvars(f), " ", nvars(g))
            term_coeff[key+g.exp[i]] = g.coeff[g.exp[i]] * elem
        end 
        # compute the exponenets of b_i T^i * f
        term_exp = [g.exp[i] + f.exp[j] for j in eachindex(f)]
        push!(summands, TropicalPuiseuxPoly(term_coeff, term_exp, true))
        #println(string(TropicalPuiseuxPoly(term_coeff, term_exp, true)))
    end 
    return quicksum(summands)
end 

function Base.:+(f::TropicalPuiseuxRational{T}, g::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    num = f.num * g.den + f.den * g.num 
    den = f.den * g.den 
    return TropicalPuiseuxRational(num, den) 
end 

# Quick addition for rational functions 
function add_with_quicksum(f::TropicalPuiseuxRational{T}, g::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    num = mul_with_quicksum(f.num, g.den) + mul_with_quicksum(f.den, g.num)
    den = mul_with_quicksum(f.den, g.den) 
    return TropicalPuiseuxRational(num, den) 
end 

# Quick multiplication for rational functions
function mul_with_quicksum(F::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    mul = TropicalPuiseuxPuiseux_one(nvars(f), f)
    for f in F 
        mul = mul_with_quicksum(mul, f)
    end 
    return mul
end 

# Quick product over vector of polynomials
function mul_with_quicksum(F::Vector{TropicalPuiseuxPoly{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    mul = TropicalPuiseuxPoly_one(nvars(F[1]), F[1])
    for f in F 
        mul = mul_with_quicksum(mul, f)
    end 
    return mul
end 

# Quick addition for vectors of rational functions
function quicksum(F::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    denoms = [f.den for f in F]
    den = mul_with_quicksum(denoms)
    summand = Vector{TropicalPuiseuxPoly{T}}()
    sizehint!(summand, length(F))
    for i in Base.eachindex(F)
        push!(summand, mul_with_quicksum([j != i ? denoms[j] : F[i].num for j in Base.eachindex(F)]))
    end 
    return TropicalPuiseuxRational(quicksum(summand), den) 
end 

# Usual multiplication for rational functions
function Base.:*(f::TropicalPuiseuxRational{T}, g::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    num = f.num * g.num 
    den = f.den * g.den 
    return TropicalPuiseuxRational(num, den)
end 

# Quick multiplication for rational functions
function mul_with_quicksum(f::TropicalPuiseuxRational{T}, g::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    num = mul_with_quicksum(f.num, g.num) 
    den = mul_with_quicksum(f.den, g.den) 
    return TropicalPuiseuxRational(num, den)
end 

# Division for vectors of rational functions
function Base.:/(f::TropicalPuiseuxRational{T}, g::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    num = f.num*g.den 
    den = f.den*g.num
    return TropicalPuiseuxRational(num, den)
end 

# Quick division for vectors of rational functions
function div_with_quicksum(f::TropicalPuiseuxRational{T}, g::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    num = mul_with_quicksum(f.num, g.den) 
    den = mul_with_quicksum(f.den, g.num)
    return TropicalPuiseuxRational(num, den)
end 

# Scalar multiplication for rational functions
function Base.:*(a::TropicalSemiringElem, f::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return TropicalPuiseuxRational(a*f.num, f.den)
end 

# Exponentiation in the tropical semiring
function Base.:^(a::TropicalSemiringElem{typeof(max)}, b::TropicalSemiringElem{typeof(max)})
    R = tropical_semiring(max)
    return R(Rational(a)*Rational(b))
end 

# Exponentiation of element of tropical semiring by rational number.
function Base.:^(a::TropicalSemiringElem{typeof(max)}, b::Rational{T}) where T<:Integer 
    R = tropical_semiring(max)
    return R(Rational(a)*b)
end

# exponentiation of a tropical Puiseux polynomial by a positive rational
function Base.:^(f::TropicalPuiseuxPoly, rat::Float64)
    if rat == 0
        return TropicalPuiseuxPoly_one(nvars(f), f)
    else 
        new_f_coeff = Dict()
        new_f_exp = copy(f.exp)
        new_f_exp = rat * new_f_exp 
        for (key, elem) in f.coeff
            new_f_coeff[rat*key] = elem^rat
        end 
        return TropicalPuiseuxPoly(new_f_coeff, new_f_exp, true)
    end 
end 

# exponentiation of a tropical Puiseux rational function by a positive rational
function Base.:^(f::TropicalPuiseuxRational, rat::Float64)
    if rat == 0
        return TropicalPuiseuxRational_one(nvars(f), f)
    else 
        return TropicalPuiseuxRational(f.num^rat , f.den^rat)
    end 
end 

function Base.:^(f::TropicalPuiseuxPoly{T}, int::Int64) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    new_f_coeff = Dict()
    new_f_exp::Vector{Vector{T}} = copy(f.exp)
    new_f_exp = int * new_f_exp 
    for (key, elem) in f.coeff
        new_f_coeff[int*key] = elem^int
    end 
    return TropicalPuiseuxPoly(new_f_coeff, new_f_exp, true)
end 

function Base.:^(f::TropicalPuiseuxPoly, int::Rational{T}) where T<:Integer
    new_f_coeff = Dict()
    new_f_exp = convert(Vector{Vector{Rational{BigInt}}}, f.exp)
    new_f_exp::Vector{Vector{Rational{BigInt}}} = Vector{Rational{BigInt}}.(int * new_f_exp) 
    for (key, elem) in f.coeff
        new_f_coeff[int*key] = elem^int
    end 
    return TropicalPuiseuxPoly(new_f_coeff, new_f_exp, true)
end 

# exponentiation of a tropical Puiseux rational function by a positive integer
function Base.:^(f::TropicalPuiseuxRational, int::Int64)
    if int == 0
        return TropicalPuiseuxRational_one(nvars(f), f)
    else 
        return TropicalPuiseuxRational(f.num^int , f.den^int)
    end 
end 

# exponentiation of a tropical Puiseux rational function by a positive integer
function Base.:^(f::TropicalPuiseuxRational, int::Rational{T}) where T<:Integer
    if int == 0
        return TropicalPuiseuxRational_one(nvars(f), f)
    else 
        return TropicalPuiseuxRational(f.num^int , f.den^int)
    end 
end 

function Base.:*(a::TropicalSemiringElem, f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    new_f_coeff = copy(f.coeff)
    new_f_exp = copy(f.exp)
    for i in eachindex(f)
        new_f_coeff[f.exp[i]] = a*f.coeff[f.exp[i]]
    end 
    return TropicalPuiseuxPoly(new_f_coeff, new_f_exp, true)
end

####################################################################

############# CODE FOR COMPOSITION ##################################

# composition in the multivariate case
function comp(f::TropicalPuiseuxPoly{T}, G::Vector{TropicalPuiseuxPoly{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    comp = TropicalPuiseuxPoly_zero(nvars(G[1]), f)
    # evaluate monomial-wise
    for (exp, coeff) in f.coeff
        term = TropicalPuiseuxPoly_one(nvars(G[1]), f)
        for i in Base.eachindex(G)
            # multiply each variable in the monomial 
            term *= G[i]^exp[i]
        end 
        # println(string(coeff * term))
        # println("term ", exp, " ", coeff)
        comp += coeff * term
    end 
    return comp
end

# multivariate case
function comp(f::TropicalPuiseuxPoly{T}, G::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    if length(G) != nvars(f)
        println("Number of variables issue")  
    else 
        comp = TropicalPuiseuxRational_zero(nvars(G[1]), G[1])
        for (key, val) in f.coeff
            term = TropicalPuiseuxRational_one(nvars(G[1]), G[1])
            for i in Base.eachindex(G)
                term *= G[i]^key[i]
            end
            comp += val * term 
        end 
        return comp
    end
end

# Quick version of composition 
function comp_with_quicksum(f::TropicalPuiseuxPoly{T}, G::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    if length(G) != nvars(f)
        println("Number of variables issue")  
    else 
        summands = Vector{TropicalPuiseuxRational{T}}()
        sizehint!(summands, length(f.exp))
        comp = TropicalPuiseuxRational_zero(nvars(G[1]), G[1])
        for (key, val) in f.coeff
            term = TropicalPuiseuxRational_one(nvars(G[1]), G[1])
            for i in Base.eachindex(G)
                term = mul_with_quicksum(term, G[i]^key[i])
            end
            push!(summands, val * term)
        end 
        return quicksum(summands)
    end
end

# Quick version of composition 
function comp_with_mul_with_quicksum(f::TropicalPuiseuxPoly{T}, G::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    if length(G) != nvars(f)
        println("Number of variables issue")  
    else 
        comp = TropicalPuiseuxRational_zero(nvars(G[1]), G[1])
        for (key, val) in f.coeff
            term = TropicalPuiseuxRational_one(nvars(G[1]), G[1])
            for i in Base.eachindex(G)
                term = mul_with_quicksum(term, G[i]^key[i])
            end
            comp = add_with_quicksum(comp, val * term)
        end 
        return comp
    end
end

# multivariate case 
function comp(f::TropicalPuiseuxRational{T}, G::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    num =  comp(f.num, G)
    den = comp(f.den, G)
    val = num / den 
    return val 
end 

# Quick version of composition 
function comp_with_quicksum(f::TropicalPuiseuxRational{T}, G::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    num =  comp_with_quicksum(f.num, G)
    den = comp_with_quicksum(f.den, G)
    val = div_with_quicksum(num, den) 
    return val 
end 

# Quick version of composition 
function comp_with_mul_with_quicksum(f::TropicalPuiseuxRational{T}, G::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    num =  comp_with_mul_with_quicksum(f.num, G)
    den = comp_with_mul_with_quicksum(f.den, G)
    val = div_with_quicksum(num, den) 
    return val 
end 

# double multivariate case 
function comp(F::Vector{TropicalPuiseuxRational{T}}, G::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return [comp(f, G) for f in F]
end 

# Quick version of composition 
function comp_with_quicksum(F::Vector{TropicalPuiseuxRational{T}}, G::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return [comp_with_quicksum(f, G) for f in F]
end 

# Quick version of composition 
function comp_with_mul_with_quicksum(F::Vector{TropicalPuiseuxRational{T}}, G::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return [comp_with_mul_with_quicksum(f, G) for f in F]
end 

#########################################################################################################

# remove all zero monomials from the expression of f
function dedup_monomials(f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    new_exp::Vector{Vector{T}} = []
    new_coeff=Dict()
    tropical_zero = zero(f.coeff[f.exp[1]])
    for i in f.exp
        if i != tropical_zero
            push!(new_exp, i)
            new_coeff[i] = f.coeff[i]
        else 
            println("found a zero")
        end
    end 
    return TropicalPuiseuxPoly(new_coeff, new_exp)
end 

function dedup_monomials(f::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return TropicalPuiseuxRational(dedup_monomials(f.num), dedup_monomials(f.den))
end 

function dedup_monomials(F::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return [dedup_monomials(f) for f in F]
end 

# Count the number of monomials appearing in a tropical expression
function monomial_count(f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return length(f.exp)
end 

# Count the number of monomials appearing in a tropical expression
function monomial_count(f::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return monomial_count(f.num) + monomial_count(f.den)
end 

# Count the number of monomials appearing in a tropical expression
function monomial_count(F::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return sum([monomial_count(f) for f in F])
end