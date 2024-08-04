using BenchmarkTools
include("../src/rat_maps.jl")

R = tropical_semiring(max)

f_coeff = [R(1), R(2)]
g_coeff = [R(2), R(1)]
h_coeff = [R(1), R(8)]
i_coeff = [R(1), R(8), R(7)]
j_coeff = [one(R)]
k_coeff = [one(R), one(R)]
l_coeff = [R(3), R(10), R(9), R(8)]
m_coeff = [R(2), R(1)]

# f = x^1 + 2x^0 = max(x+1, 2)
f_exp::Vector{Vector{Rational{BigInt}}} = [[1.0], [0.0]]
g_exp::Vector{Vector{Rational{BigInt}}} = [[1.0], [0.0]]
h_exp::Vector{Vector{Rational{BigInt}}} = [[1.2, 3.9], [4.8, 1.7]]
i_exp::Vector{Vector{Rational{BigInt}}} = [[0.0], [1.0], [2.0]]
j_exp::Vector{Vector{Rational{BigInt}}} = [[0.0]]
k_exp::Vector{Vector{Rational{BigInt}}} = [[1.0, 1.0], [1.0, 0.0]]
l_exp::Vector{Vector{Rational{BigInt}}} = [[0.0],[1.0], [2.0], [3.0]]
m_exp::Vector{Vector{Rational{BigInt}}} = [[1.0], [0.0]]


f = TropicalPuiseuxPoly(f_coeff, f_exp)
g = TropicalPuiseuxPoly(g_coeff, g_exp)
h = TropicalPuiseuxPoly(h_coeff, h_exp)
i = TropicalPuiseuxPoly(i_coeff, i_exp)
j = TropicalPuiseuxPoly(j_coeff, j_exp)
k = TropicalPuiseuxPoly(k_coeff, k_exp)
l = TropicalPuiseuxPoly(l_coeff, l_exp)
m = TropicalPuiseuxPoly(m_coeff, m_exp)

F = TropicalPuiseuxRational(f, g)
G = TropicalPuiseuxRational(i, j)
H = TropicalPuiseuxRational(l, m)
K = comp(k, [F, G])

a = [R(0)]
b = [R(-8)]
c = [R(53//10)]
d = [R(-90//18)]

#### ADD #####
# @show eval(f+g, a) 
# @show eval(f, a) + eval(g, a)
# @show eval(f+g, b) 
# @show eval(f, b) + eval(g, b)

#### MUL #####
# @show eval(f*g, a) 
# @show eval(f, a) * eval(g, a)
# @show eval(f*g, b) 
# @show eval(f, b) * eval(g, b)


#### COMP #####
# @show eval(k, [eval(F, b), eval(G, b)])
# @show eval(K, b)
# @show eval(H, b)
# @show eval(k, [eval(F, c), eval(G, c)])
# @show eval(K, c)
# @show eval(H, c)
# @show eval(k, [eval(F, d), eval(G, d)])
# @show eval(K, d)
# @show eval(H, d)
# @show string(K)


#println(length(K.num.coeff) == length(K.num.exp))


# F = TropicalPuiseuxRational(f, TropicalPuiseuxPoly_one(nvars(f), f))
# @show comp(f, [f])
# @show comp(x_mul_one, [f])

# @show eval(f, a)
# @show eval(F, eval(F, b))
# # comp(f, f) = max(max(x+1, 2)+1, 2) = comp(1x + 2, 1x + 2) = (1*1)x + 1*2 + 2 = 2x+3
# @show eval(comp(F, [F]), b)

function random_TropicalRatioanl(n_var, n_terms)
    num_exp = Vector{Vector{Rational{BigInt}}}()
    den_exp = Vector{Vector{Rational{BigInt}}}()
    num_coeff = Dict()
    den_coeff = Dict()
    #println("here1")
    for i in 1:n_terms
        new_num_exp = Rational{BigInt}.(rand(n_var))
        new_den_exp = Rational{BigInt}.(rand(n_var))
        push!(num_exp, new_num_exp)
        push!(den_exp, new_den_exp)
        num_coeff[new_num_exp] = R(Rational{BigInt}.(rand()))
        den_coeff[new_den_exp] = R(Rational{BigInt}.(rand()))
        #println("here2")
    end 
    num = TropicalPuiseuxPoly(num_coeff, num_exp)
    den = TropicalPuiseuxPoly(den_coeff, den_exp)
    return TropicalPuiseuxRational(num, den)
end 

function random_TropicalPoly(n_var, n_terms)
    num_exp = Vector{Vector{Rational{BigInt}}}()
    num_coeff = Dict()
    #println("here1")
    for i in 1:n_terms
        new_num_exp = Rational{BigInt}.(rand(n_var))
        push!(num_exp, new_num_exp)
        num_coeff[new_num_exp] = R(Rational{BigInt}.(rand()))
    end 
    num = TropicalPuiseuxPoly(num_coeff, num_exp)
    return num 
end 

# for i in 1:10
#     #println("here, " ,i)
#     f = random_TropicalRatioanl(2, 3)
#     g = random_TropicalRatioanl(2, 3)
#     @show monomial_count(f)
#     @show monomial_count(g)
#     k = random_TropicalRatioanl(2, 2)
#     a = Rational{BigInt}.(rand(1))
#     aa = [R(a_i) for a_i in a]
#     #println("Computing first operation")
#     cp = comp(k, [f, g])
#     @show (eval(cp, aa)) == eval(k, [eval(f, aa), eval(g, aa)])
#     #println("Computing second operation")
# end 

KL =  random_TropicalPoly(2, 3)
a = Rational{BigInt}.(rand(1))
aa = [R(a_i) for a_i in a]
@show eval(KL + KL + KL, aa) == eval(quicksum([KL, KL, KL]), aa)
@show eval(KL * KL, aa) == eval(mul_with_quicksum(KL, KL), aa)

function test_sum(n_vars, n_terms, n_summands)
    KL = [random_TropicalPoly(n_vars, n_terms) for i in 1:n_summands]
    return sum(KL)
end 

function test_quicksum(n_vars, n_terms, n_summands)
    KL = [random_TropicalPoly(n_vars, n_terms) for i in 1:n_summands]
    return quicksum(KL)
end 