include("../src/linear_regions.jl")
include("../src/mlp_to_trop.jl")

### UNIT TESTS ####

## Tropical Rational maps ###

R = tropical_semiring(max) 

# k is the tropical polynomial 0x+0y+0xy = max{x, y, x+y}. This has 
k = TropicalPuiseuxPoly([0.0, 0.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
# l is the tropical polynomial 0x^0y^0 = 0
l = TropicalPuiseuxPoly([0.0], [[0.0, 0.0]])
# The number of linear regions of k/l is the number of linear regions of l, i.e. 1.
println(length(enum_linear_regions_rat(k, l)))

# Now run checks for the case where cancelation happens:

# f = 0x^1 + x^1y^1
f = TropicalPuiseuxPoly([R(0), R(0)], [[1.0, 0.0], [1.0, 1.0]])
# g = 0.0x^4y^3 + 0.0x^4y^4
g = TropicalPuiseuxPoly([R(0), R(0)], [[4.0, 3.0], [4.0, 4.0]])
# f/g has precisely 1 linear region.
println(length(enum_linear_regions_rat(f, g)))

m = TropicalPuiseuxPoly([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
n = TropicalPuiseuxPoly([0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]])

a = [R(6//11), R(3//8)]

k = TropicalPuiseuxRational(f, g)

#@show eval_temporary(k, [eval_temporary(k, a), eval_temporary(k, a)])
#@show eval_temporary(comp(k, [k, k]), a)

### MLP and Tropical maps #####

function mlp_eval(weights, biases, thresholds, a)
    # feed-forward function.
    if length(a) != size(weights[1], 2)
        println("Input dimension is incorrect...")
    end 
    output = Rational.(aa)
    for i in Base.eachindex(weights)
        output = weights[i]*output + Rational.(biases[i])
        for j in Base.eachindex(output)
            output[j] = max(output[j], Rational(thresholds[i][j]))
        end
    end 
    return output
end 

R = tropical_semiring(max) 

w, b, t = random_mlp([3, 3, 2, 2, 2])
#@show w, b, t = random_mlp([3, 1])

#A = [1.0  0.0  1.0]
#b = [1.0]
#t = [0.0]

a = [0, 1, 1]

#trop = single_to_trop(w[1], b[1], t[1])
trop = mlp_to_trop(w, b, t)

aa = [R(0), R(1), R(1)]

@show mlp_eval(w, b, t, a)
@show [eval(i, aa) for i in trop]