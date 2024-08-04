using BenchmarkTools
include("../src/rat_maps.jl")

R = tropical_semiring(max)

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
        new_num_exp = Rational{BigInt}.(randn(n_var))
        push!(num_exp, new_num_exp)
        num_coeff[new_num_exp] = R(Rational{BigInt}.(rand()))
    end 
    num = TropicalPuiseuxPoly(num_coeff, num_exp)
    return num 
end 

# for i in 1:10
#     #println("here, " ,i)
#     #f = random_TropicalPoly(2, 3)
#     #g = random_TropicalPoly(2, 3)
#     f = random_TropicalRatioanl(2, 3)
#     g = random_TropicalRatioanl(2, 3)
#     # @show monomial_count(f)
#     # @show monomial_count(g)
#     # k = random_TropicalRatioanl(2, 2)
#     a = Rational{BigInt}.(rand(1))
#     aa = [R(a_i) for a_i in a]
#     # #println("Computing first operation")
#     # cp = comp_with_quicksum(k, [f, g])
#     # @show (eval(cp, aa)) == eval(k, [eval(f, aa), eval(g, aa)])
#     #println("Computing second operation")
#     @show eval(quicksum([f, g]), aa) == eval(f, aa) + eval(g, aa) 
#     #@show eval(mul_with_quicksum(f, g), aa) == eval(f, aa) * eval(g, aa)
# end 

KL =  random_TropicalPoly(4, 5)
KL2 =  random_TropicalPoly(4, 3)
KL3 =  random_TropicalPoly(4, 8)
a = Rational{BigInt}.(rand(1))
aa = [R(a_i) for a_i in a]
@show eval(KL + KL2 + KL3, aa) == eval(quicksum([KL, KL2, KL3]), aa)
@show eval(KL * KL2, aa) == eval(mul_with_quicksum(KL, KL2), aa)