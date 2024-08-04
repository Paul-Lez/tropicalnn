using BenchmarkTools
include("../src/linear_regions.jl")
include("../src/mlp_to_trop.jl")
include("../src/mlp_to_trop_with_elim.jl")

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
    local w, b, t = random_mlp([3, 3, 2, 1])

    local trop0 = mlp_to_trop(w, b, t)
    local trop1 = mlp_to_trop_with_elim(w, b, t)

    local R = tropical_semiring(max) 

    local a = Rational{BigInt}.(rand(3))
    local aa = [R(a_i) for a_i in a]
    @show (R(mlp_eval(w, b, t, a)[1]) == eval(trop0, aa)[1])
end