using BenchmarkTools
include("../src/linear_regions.jl")

R = tropical_semiring(max) 

# f = x^2+y^1 + x^1y^2
f = TropicalPuiseuxPoly([R(0), R(0)], [[2.0, 1.0], [1.0, 2.0]])
# g = x^1y^0 + x^0y^1
g = TropicalPuiseuxPoly([R(0), R(0)], [[1.0, 0.0], [0.0, 1.0]])
lin_f = enum_linear_regions(f)
lin_g = enum_linear_regions(g)
# f/g has precisely 1 linear region.
println(length(enum_linear_regions_rat(f, g)))
println(lin_f)
println(lin_g)



# m = TropicalPuiseuxPoly([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
# n = TropicalPuiseuxPoly([0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]])

# a = [R(6//11), R(3//8)]

# k = TropicalPuiseuxRational(f, g)
