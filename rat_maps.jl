using Pkg 
Pkg.add("Oscar")
#Pkg.add("StatsBase")
Pkg.add("Combinatorics")
using Oscar
#using StatsBase
using Combinatorics

struct TropicalPuissieuxPoly
    coeff::Vector
    exp::Vector
end 

function eachindex(f::TropicalPuissieuxPoly)
    return Base.eachindex(f.coeff)
end 

function n_var(f::TropicalPuissieuxPoly)
    if !is_empty(f.coeff)
        return length(f.exp[1])
    else 
        return -1
    end 
end 

function polyhedron(f::TropicalPuissieuxPoly, i)
    A = Base.reduce(vcat, [f.exp[j] - f.exp[i] for j in eachindex(f)]')
    b = [f.coeff[i] - f.coeff[j] for j in eachindex(f)]
    return Oscar.polyhedron(A, b)
end 

# return a list of the linear regions of f and a list of the terms of f that are never attained
function enum_linear_regions(f::TropicalPuissieuxPoly)
    linear_regions = []
    for i in eachindex(f)
        poly = polyhedron(f, i)
        push!(linear_regions, (poly, Oscar.is_feasible(poly)))
    end 
    return linear_regions
end 

#implement ignoring functionality
function check_linear_repetitions(f::TropicalPuissieuxPoly, g::TropicalPuissieuxPoly)
    lin_f = enum_linear_regions(f)
    #println(lin_f)
    lin_g = enum_linear_regions(g)
    linear_map = Dict()
    for i in eachindex(f)
        for j in eachindex(g)
            if lin_f[i][2] && lin_g[j][2]
                poly = Oscar.intersect(lin_f[i][1], lin_g[j][1])
                #println(Oscar.dim(poly), " here")
                #println(Oscar.is_feasible(poly))
                if Oscar.is_feasible(poly) && Oscar.dim(poly) == n_var(f)
                    #println("here")
                    linear_map[(i, j)] = [f.coeff[i] - g.coeff[j], f.exp[i] - g.exp[j]]
                end 
            #push!(linear_map, [f.coeff[i] - g.coeff[j], f.exp[i] - g.exp[j]])
            end 
        end 
    end
    #println(linear_map)
    linear_map_unique = unique([l for (key, l) in linear_map])
    if length(linear_map) == length(linear_map_unique)
        return linear_map, Dict(), false
    else 
        #print("here")
        # compute indices of repetitions for each linear map
        reps = [(l, Base.findall(x -> x == l, linear_map)) for l in linear_map_unique]
        #print(reps)
        #print("here")
        # reps' is the dictionary indexed by linear maps, where the entry for l is the indices of repetitions in the array `linear_map`
        #reps' = Dict(reps)
        #= # this can probably be made more efficient...
        for i in linear_map_unique 
            ith_reps = []
            for j in Base.eachindex(linear_map)
                if linear_map[j] == i 
                    append!(ith_reps, j)
                end 
            end 
            append!(reps, ith_reps)
        end  =#
        return linear_map, reps, true 
    end
end 

#= function rat_linear_partition(f::TropicalPuissieuxPoly, g::TropicalPuissieuxPoly)
    lin_f, ignore_f = enum_linear_regions(f)
    lin_g, ignore_g = enum_linear_regions(g)
    return [Oscar.intersect(p1, p2) for (_, p1) in lin_f for (_, p2) in lin_g]
end  =#

function connected_closure(V, D)
    #visited = Dict(V .=> false)
    comp = Dict(V .=> V)
    for v in V 
        for w in V 
            if v == w || (haskey(D, (v, w)) && D[(v, w)]) || (haskey(D, (w, v)) && D[(w, v)]) 
                comp[w] = comp[v]
            end
        end 
    end 
    return unique([v for (_, v) in comp])
end

function enum_linear_regions_rat(f::TropicalPuissieuxPoly, g::TropicalPuissieuxPoly)
    lin_f = enum_linear_regions(f)
    lin_g = enum_linear_regions(g)
    linear_maps, reps, exists_reps = check_linear_repetitions(f, g)
    lin_regions = []
    if exists_reps 
        has_intersect = Dict()
        # first find all pairwise intersections of polytopes.
        #println(reps)
        for (_, vals) in reps 
            if length(vals) == 1 
                append!(lin_regions, vals)
            else
                #println(vals)
                for ((i, j), (k, l)) in combinations(vals, 2)
                    #println(i, j, k, l)
                    poly11, _ = lin_f[i]
                    poly12, _ = lin_g[j]
                    poly21, _ = lin_f[k]
                    poly22, _ = lin_g[l]
                    poly1 = Oscar.intersect(poly11, poly12)
                    poly2 = Oscar.intersect(poly21, poly22)
                    int = Oscar.intersect(poly1, poly2)
                    has_intersect[((i, j), (k, l))] = Oscar.is_feasible(int)
                end
                # now find transitive closure of the relation given by dictionary has_intersect.
                append!(lin_regions, connected_closure(vals, has_intersect))
            end 
        end
        return lin_regions 
    else 
        # if there are no repetitions, then the linear regions are just the intersections of the linear regions of f and the linear regions of g
        return [Oscar.intersect(p1, p2) for (p1, _) in lin_f for (p2, _) in lin_g]
    end
end 

### UNIT TESTS ####
# f = 0x + 1x^2 + 2x^5
#f = TropicalPuissieuxPoly([0.0, 1.0, 2.0], [[1.0, 0.0, 8.1, 8.1], [2.0, 5.0, 8.8, 9.9], [5.0, 8.0, 1.1, 0.1]])
#g = TropicalPuissieuxPoly([1.0, 1.0, 2.0], [[0.0, 0.0, 11.0, 17.0], [1.0, 8.0, 8.8, 9.0], [4.0, 11.0, 0.5, 8.1]])
#f = TropicalPuissieuxPoly([1.0, 1.0, 1.0], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

#println(polyhedron(f, 2))
#println(enum_linear_regions(f))
#println(check_linear_repetitions(f, g, 1, 1))
#println(length(enum_linear_regions_rat(f, g)))

n_terms = 50
n_variables = 28

for i in 1:2
    c_f = rand(Float64, n_terms)
    c_g = rand(Float64, n_terms)
    exp_f = []
    exp_g = []
    for j in 1:n_terms 
        push!(exp_f, rand(Float64, n_variables))
        push!(exp_g, rand(Float64, n_variables))
    end
    #println(exp_f)
    f = TropicalPuissieuxPoly(c_f, exp_f)
    g = TropicalPuissieuxPoly(c_g, exp_g)
    println(length(enum_linear_regions_rat(f, g)))
end 