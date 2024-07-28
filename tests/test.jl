using BenchmarkTools
include("../src/linear_regions.jl")
include("../src/mlp_to_trop.jl")
include("../src/mlp_to_trop_with_elim.jl")

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

N_samples = 10
avg_red = 0
avg = 0

for i in 1:N_samples
    println("Sample ", i)
    w, b, t = random_mlp([3, 2, 2, 2, 1])
    #@show w, b, t = random_mlp([3, 1])

    #A = [1.0  0.0  1.0]
    #b = [1.0]
    #t = [0.0]

    a = [0, 1, 1]

    #trop = single_to_trop(w[1], b[1], t[1])
    println("computing tropical form of random MLP")
    t1 = time()
    trop1 = mlp_to_trop_with_elim(w, b, t)
    t2 = time()
    #trop2 = mlp_to_trop_with_dedup(w, b, t)
    t3 = time()
    #trop3 = monomial_elim(trop2)
    #t4 = time()
    #trop4 = mlp_to_trop_with_quasi_elim(w, b, t)
    t5 = time()
    trop5 = mlp_to_trop_with_double_elim(w, b, t)
    t6 = time()

    println("Computation with elim took ", t2 - t1)
    #println("Computation (vanilla) took ", t3 - t2)
    #println("Computation with one-shot elim took ", t4 - t2)
    #println("Computation with quasi-elim took ", t5 - t4)
    println("Computation with double-elim took ", t6 - t5)
    
    n_red =  monomial_count(trop1)
    #n =  monomial_count(trop2)
    println("Reduced form ", n_red)
    #println("Non reduced form ", n)
    #println("Reduced form, one-shot approach", monomial_count(trop3))
    #println("Reduced form, quasi-elim approach ", monomial_count(trop4))
    println("Reduced form, double-elim approach ", monomial_count(trop5))
    #global avg_red += n_red / N_samples
    #global avg += n / N_samples

    println("_________________________________")
end 

#println(avg_red, "    ", avg)


#aa = [R(0), R(1), R(1)]

#@show mlp_eval(w, b, t, a)
#@show [eval(i, aa) for i in trop]

function test_rat_maps()

    R = tropical_semiring(max)

    f_coeff = [R(1), R(2)]
    g_coeff = [R(2), R(1)]
    h_coeff = [R(1), R(8)]
    i_coeff = [R(1), R(8), R(7)]
    j_coeff = [one(R)]
    f_exp = [[1.0], [0.0]]
    g_exp = [[1.0], [0.0]]
    h_exp = [[1.2, 3.9], [4.8, 1.7]]
    i_exp = [[0.0], [1.0], [2.0]]
    j_exp = [[0.0]]

    f = TropicalPuiseuxPoly(f_coeff, f_exp)
    g = TropicalPuiseuxPoly(g_coeff, g_exp)
    h = TropicalPuiseuxPoly(h_coeff, h_exp)
    i = TropicalPuiseuxPoly(i_coeff, i_exp)
    j = TropicalPuiseuxPoly(j_coeff, j_exp)

    #println(string(f*j))
    println(string(j*f))
    #println(string(f^4))
    #println(string(comp(i, f / g)))
    #println(string(f/g + f/g))
    #println(string(i+f))
    #println(string(g+i))
    #println(string(f*TropicalPuiseuxPoly_one(1, f)), " = ", string(f))

    F = TropicalPuiseuxRational(f, g)
    println(string(comp(h, [F, F])))
end 

######### Tests #############
function test_mlp_to_trop()
    R = tropical_semiring(max)
    A1 = [[1.0, 1.0],
        [2.0, 2.2 ]]

    A2 = [[1], [2]]

    b1 = [R(1),
        R(2)]

    b2 = [R(1)]

    t1 = [R(2), 
        R(2)]

    t2 = [R(0)]

    #println(single_to_trop(A1, b1, t1))
    #p = mlp_to_trop([A1, A2], [b1, b2], [t1, t2])[1]
    #println(string(p))

    d1 = 5
    d2 = 2
    d3 = 2
    d4 = 10
    d5 = 1

    A1 = rand(d1, d2)
    A2 = rand(d2, d3)
    A3 = rand(d3, d4)
    A4 = rand(d4, d5)

    b1 = [R(QQ(Rational(i))) for i in rand(Float32, d2)]
    b2 = [R(QQ(Rational(i))) for i in rand(Float32, d3)]
    b3 = [R(QQ(Rational(i))) for i in rand(Float32, d4)]
    b4 = [R(QQ(Rational(i))) for i in rand(Float32, d5)]

    t1 = [R(QQ(Rational(i))) for i in rand(Float32,d2)]
    t2 = [R(QQ(Rational(i))) for i in rand(Float32, d3)]
    t3 = [R(QQ(Rational(i))) for i in rand(Float32, d4)]
    t4 = [R(QQ(Rational(i))) for i in rand(Float32, d5)]

    p = mlp_to_trop([A1, A2, A3, A4], [b1, b2, b3, b4], [t1, t2, t3, t4])[1] #+ mlp_to_trop([A1, A2, A3, A4], [b1, b2, b3, b4], [t1, t2, t3, t4])[2]
    arr1 = copy(p.num.exp)
    arr2 = copy(p.den.exp)
    println(length(arr1))
    println(length(arr2))
    filter!(e -> p.num.coeff[e] != zero(R), arr1)
    filter!(e -> p.den.coeff[e] != zero(R), arr2)
    println(length(arr1))
    println(length(arr2))

    weights, bias, thresholds = random_mlp([5, 5, 2], true)
    trop = mlp_to_trop(weights, bias, thresholds)
    println("done")
    #println(trop)
    #println(weights)
    #@show length(trop)
    R = tropical_semiring(max)

    A1 = [3.0 -5.0;
            7.0 8.0]

    b1 = [R(3),
        R(7)]

    t1 = [ R(0),
        R(0)]

    g = single_to_trop(A1, b1, t1)

    for i in g
        println(string(i))
    end 
end 