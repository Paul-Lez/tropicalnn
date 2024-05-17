
include("linear_regions.jl")

### UNIT TESTS ####

# k is the tropical polynomial 0x+0y+0xy = max{x, y, x+y}. This has 
k = TropicalPuiseuxPoly([0.0, 0.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
# l is the tropical polynomial 0x^0y^0 = 0
l = TropicalPuiseuxPoly([0.0], [[0.0, 0.0]])
# The number of linear regions of k/l is the number of linear regions of l, i.e. 1.
println(length(enum_linear_regions_rat(k, l)))

# Now run checks for the case where cancelation happens:

# f = 0x^1 + x^1y^1
f = TropicalPuiseuxPoly([0.0, 0.0], [[1.0, 0.0], [1.0, 1.0]])
# g = 0.0x^4y^3 + 0.0x^4y^4
g = TropicalPuiseuxPoly([0.0, 0.0], [[4.0, 3.0], [4.0, 4.0]])
# f/g has precisely 1 linear region.
println(length(enum_linear_regions_rat(f, g)))

m = TropicalPuiseuxPoly([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
n = TropicalPuiseuxPoly([0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]])
