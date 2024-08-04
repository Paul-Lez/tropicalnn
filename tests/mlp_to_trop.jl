using BenchmarkTools
using BenchmarkTools
include("../src/linear_regions.jl")
include("../src/mlp_to_trop.jl")

function mlp_eval(weights, biases, thresholds, a)
    # feed-forward function.
    if length(a) != size(weights[1], 2)
        println("Input dimension is incorrect...")
    end 
    output::Vector{Rational{BigInt}} = Rational.(a)
    for i in Base.eachindex(weights)
        output = weights[i]*output + Rational.(biases[i])
        for j in Base.eachindex(output)
            output[j] = max(output[j], Rational(thresholds[i][j]))
        end
    end 
    return output
end 

N_test = 20

for i in 1:N_test
    local w, b, t = random_mlp([100, 8, 1])

    @time local trop0 = mlp_to_trop(w, b, t)
    @time local trop1 = mlp_to_trop_with_quicksum(w, b, t)#mlp_to_trop_with_mul_with_quicksum
    @time local trop1 = mlp_to_trop_with_mul_with_quicksum(w, b, t)
    #@show monomial_count(trop1)

    local R = tropical_semiring(max) 

    local a = Rational{BigInt}.(rand(100))
    local aa = [R(a_i) for a_i in a]
    @show (R(mlp_eval(w, b, t, a)[1]) == eval(trop1, aa)[1])
end

# w, b, t = random_mlp([100, 30, 1])
# @time trop1 = mlp_to_trop_with_quicksum(w, b, t)