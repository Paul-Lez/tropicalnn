using Pkg 
using Oscar
using Combinatorics
using Base.Threads
include("rat_maps.jl")


function polyhedron(f::TropicalPuiseuxPoly, i)
    # take A to be the matrix as in the overleaf document
    A = mapreduce(permutedims, vcat, [f.exp[j] - f.exp[i] for j in eachindex(f)])
    # and ditto for b
    b = [f.coeff[f.exp[i]] - f.coeff[j] for j in f.exp]
    # return the corresponding linear region
    return Oscar.polyhedron(A, b)
end 

# return a list of the linear regions of f
function enum_linear_regions(f::TropicalPuiseuxPoly)
    """
    enum_linear_regions(f) outputs an array of tuples (poly, bool) indexed by the same set as the exponents of f. poly is the linear region corresponding to the exponent, and bool is true when this region is nonemtpy.
    """
    linear_regions = Vector()
    sizehint!(linear_regions, length(f.exp))
    for i in eachindex(f)
        poly = polyhedron(f, i)
        # add the polyhedron to the list plus a bool saying whether the polyhedron is non-empty
        push!(linear_regions, (poly, Oscar.is_feasible(poly)))
    end 
    return linear_regions
end 

function connected_closure(V, D)
    """
    Given a list of vertices V and a boolean dictionary indexed by pairs of vertices, outputs a list of representatives 
    for equivalence classes by the closure of D (i.e. the smallest equivalence relation D)
    """
    # comp is a dictionary that assigns to each vertex v a vertex v' with the property that if v ~ w then v' = w'
    comp = Dict(V .=> V)
    for v in V 
        for w in V 
            # TODO Paul: check that this update rule is actually the right one. Do we need to keep track of vertices that have already been visited?
            if v == w || (haskey(D, (v, w)) && D[(v, w)]) || (haskey(D, (w, v)) && D[(w, v)]) 
                comp[w] = comp[v]
            end
        end 
    end 
    return unique([v for (_, v) in comp])
end

#TODO Paul: work out what we want to have as output when there are repetitions
function enum_linear_regions_rat(f::TropicalPuiseuxPoly, g::TropicalPuiseuxPoly, verbose=false)
    """
    Should output the linear regions of a tropical Puiseux rational function. 

    """
    # first, compute the linear regions of f and g. 
    if verbose
        println("Computing linear regions of f and g")
    end 
    lin_f = enum_linear_regions(f)
    lin_g = enum_linear_regions(g)
    # next, check which for repetitions of the linear map corresponding to f/g on intersections of the linear regions computed above.
    function check_linear_repetitions()
        if verbose 
            println("Computing non-empty intersections of linear regions")
        end 
        linear_map = Dict()
        # We need to check for pairwise intersection of each polytope, by iterating over
        Threads.@threads for i in eachindex(f)
            Threads.@threads for j in eachindex(g)
                # we only need to do the checks on linear regions that are attained by f and g
                if lin_f[i][2] && lin_g[j][2]
                    # check if the polytopes intersect
                    #######t1 = time()
                    poly = Oscar.intersect(lin_f[i][1], lin_g[j][1])
                    ###########t2 = time()
                    #########println(t2-t1)
                    # if they intersect on a large enough region then add this to the list of linear maps that arise in f/g
                    if Oscar.is_feasible(poly) && Oscar.dim(poly) == nvars(f)
                        Threads.@inbounds linear_map[poly] = [f.coeff[f.exp[i]] - g.coeff[g.exp[j]], f.exp[i] - g.exp[j]]
                    end 
                end 
            end 
        end
        if verbose
            println("Checking for repetitions of linear maps")
        end 
        # check for repetitions
        linear_map_unique = unique([l for (key, l) in linear_map])
        if length(linear_map) == length(linear_map_unique)
            return linear_map, [], false
        else 
            # compute indices of repetitions for each linear map
            reps = [(l, Base.findall(x -> x == l, linear_map)) for l in linear_map_unique]
            return linear_map, reps, true 
        end
    end 
    linear_map, reps, exists_reps = check_linear_repetitions()
    # if there are no repetitions, then the linear regions are just the non-empty intersections of linear regions of f and linear regions of g
    if !exists_reps   
        lin_regions = collect(keys(linear_map))
    # if there are repetitions then we will need to find connected components of the union of the polytopes on which repetitons occur.
    else
        if verbose 
            println("Computing connected components for repeated linear maps")
        end 
        # Initialise the array lin_regions. This will contain the true linear regions of f/g
        lin_regions = []
        # first find all pairwise intersections of polytopes.
        for (_, vals) in reps 
            # if vals has length 1 then there is no other linear region with the same linear map
            if length(vals) == 1 
                append!(lin_regions, vals)
            else
                # otherwise, we check for intersections in the set of linear regions with a given map
                has_intersect = Dict()
                # iterate over unordered pairs of (distinct) elements of vals
                for (poly1, poly2) in combinations(vals, 2)
                    # intersect the two polyhedra
                    intesection = Oscar.intersect(poly1, poly2)
                    # add true to the dictionary if the intersection is nonemtpy as false otherwise
                    has_intersect[((i, j), (k, l))] = Oscar.is_feasible(intesection)
                end
                # now find transitive closure of the relation given by dictionary has_intersect.
                append!(lin_regions, connected_closure(vals, has_intersect))
            end 
        end
    end
    if verbose 
        println("The number of linear regions of the rational function is ", length(lin_regions))
    end 
    return lin_regions
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