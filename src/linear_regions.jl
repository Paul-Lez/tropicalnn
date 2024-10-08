using Pkg 
using Oscar
using Combinatorics
using Base.Threads
include("rat_maps.jl")

function to_Float64(v::Vector)
    return [Float64(w) for w in v]
end 

# Outputs the polyhedron of the i-th monomial of a tropical polynomial 
function polyhedron(f::TropicalPuiseuxPoly, i)
    """
    Input: tropical Puiseux polynomial f and index i 
    Ouput: polyhedron corresponding to points where f is given by the the linear map corresponding to the i-th monomial of f.
    """
    # take A to be the matrix as in the overleaf document
    A = mapreduce(permutedims, vcat, [to_Float64(f.exp[j]) - to_Float64(f.exp[i]) for j in eachindex(f)])
    # and ditto for b. First however we need to convert elements of b to Floats since they are elements are the tropical semiring.
    b = [Float64(Rational(f.coeff[f.exp[i]])) - Float64(Rational(f.coeff[j])) for j in f.exp]
    # return the corresponding linear region
    try 
        return Oscar.polyhedron(A, b)
    catch e
        println(A)
        println(b)
        println(e)
    end 
end 

# return a list of the linear regions of a tropical polynomial f
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
            if v == w || (haskey(D, (v, w)) && D[(v, w)]) || (haskey(D, (w, v)) && D[(w, v)]) 
                comp[w] = comp[v]
            end
        end 
    end 
    return unique([v for (_, v) in comp])
end

# Computes the number of equivalence classes of the transitive closure of a 
# relation by running depth first search
function n_components(V, D)
    count = 0
    visited = Dict()
    for v in V 
        visited[v] = false
    end 
    function depth_first_search(k)
        visited[k] = true
        for p in V
            #println(p)
            if !visited[p] && ((haskey(D, (k, p)) && D[(k, p)]) || (haskey(D, (p, k)) && D[(p, k)]))
                depth_first_search(p)
            end 
        end 
    end 
    for p in V
        if !visited[p]
            depth_first_search(p)
            count += 1
        end 
    end 
    return count
end 

# Compute the number of linear regions of a tropical rational function
function enum_linear_regions_rat(f::TropicalPuiseuxPoly, g::TropicalPuiseuxPoly, verbose=false)
    """
    Computes the number of linear regions of a tropical Puiseux rational function f/g
    Inputs: Tropical Puiseux polynomials f and g 
            boolean verbose
    Ouput: array containing linear regions of f/g represented by polyhedra. In the case where 
            the linear region is a union of such polyhedra, we pick one of the polyhedra in the union to represent this.
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
        #Threads.@threads 
        for i in eachindex(f)
            #Threads.@threads 
            for j in eachindex(g)
                # we only need to do the checks on linear regions that are attained by f and g
                if lin_f[i][2] && lin_g[j][2]
                    # check if the polytopes intersect
                    poly = Oscar.intersect(lin_f[i][1], lin_g[j][1])
                    # if they intersect on a large enough region then add this to the list of linear maps that arise in f/g
                    # Note: we used to check that the poly is feasible and has dimension n, but it's a lot fast to check that it is full dimensional directly.
                    if Oscar.is_fulldimensional(poly)     # Oscar.is_feasible(poly) && Oscar.dim(poly) == nvars(f)
                        #Threads.@inbounds 
                        linear_map[poly] = [Rational(f.coeff[f.exp[i]]) - Rational(g.coeff[g.exp[j]]), f.exp[i] - g.exp[j]]
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
    # if there are repetitions then we will need to find connected components of the union of the polytopes on which repetitions occur.
    else
        println("Linear repetition mode")
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
                    # add true to the dictionary if the intersection is nonemtpy and false otherwise
                    has_intersect[(poly1, poly2)] = Oscar.is_feasible(intesection)
                end
                # now find transitive closure of the relation given by dictionary has_intersect.
                num_components = n_components(vals, has_intersect)
                ###########################################
                # Temporary: we're only interested in the number of connected components for the experiments rather than the 
                # exact polyhedra, so instead of returning the data of the polyhedra, we just return an arbitrary list 
                # that has length the number of connected components. This is messy but means making the connected 
                # closure search a bit easier.
                # TODO: clean this up. 
                linear_regions_repeat = repeat([1], num_components)
                append!(lin_regions, linear_regions_repeat)
            end 
        end
    end
    if verbose 
        println("The number of linear regions of the rational function is ", length(lin_regions))
    end 
    return lin_regions
end 