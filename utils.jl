using TropicalNN
using Plots
using Plots.PlotMeasures
using Oscar
using LinearAlgebra

function _hrep_vertices_2d(A, b; tol=1e-8)
    size(A, 2) == 2 ||
        throw(ArgumentError("plot_linear_regions only supports 2D regions, got $(size(A, 2))D"))

    points = Vector{Vector{Float64}}()
    for i in 1:(size(A, 1) - 1)
        for j in (i + 1):size(A, 1)
            M = [A[i, 1] A[i, 2]; A[j, 1] A[j, 2]]
            abs(det(M)) <= tol && continue

            x = M \ [b[i], b[j]]
            if all(A * x .<= b .+ tol)
                push!(points, x)
            end
        end
    end

    unique_points = Vector{Vector{Float64}}()
    for point in points
        if !any(existing -> norm(existing - point, Inf) <= tol, unique_points)
            push!(unique_points, point)
        end
    end

    length(unique_points) >= 3 || return nothing

    cx = sum(point[1] for point in unique_points) / length(unique_points)
    cy = sum(point[2] for point in unique_points) / length(unique_points)
    perm = sortperm([atan(point[2] - cy, point[1] - cx) for point in unique_points])
    return unique_points[perm]
end

function _plot_highs_region!(p, poly, bbox_A, bbox_b, color)
    A = vcat(poly.A, bbox_A)
    b = vcat(poly.b, bbox_b)
    vertices = _hrep_vertices_2d(A, b)
    vertices === nothing && return p

    x_coords = [vertex[1] for vertex in vertices]
    y_coords = [vertex[2] for vertex in vertices]
    plot!(p, x_coords, y_coords, seriestype=:shape, color=color, alpha=0.6,
        linecolor=:black, linewidth=1)
    return p
end

function _plot_oscar_region!(p, poly, bbox, color)
    bounded_poly = Oscar.intersect(poly, bbox)

    if Oscar.is_feasible(bounded_poly) && Oscar.is_fulldimensional(bounded_poly)
        verts = collect(Oscar.vertices(bounded_poly))
        isempty(verts) && return p

        x_coords = [Float64(v[1]) for v in verts]
        y_coords = [Float64(v[2]) for v in verts]

        cx = sum(x_coords) / length(x_coords)
        cy = sum(y_coords) / length(y_coords)
        angles = atan.(y_coords .- cy, x_coords .- cx)
        perm = sortperm(angles)

        plot!(p, x_coords[perm], y_coords[perm], seriestype=:shape, color=color,
            alpha=0.6, linecolor=:black, linewidth=1)
    end

    return p
end

"""
    lp_mode()

Pick the `LinearRegionsCalculationMode` to use for region enumeration,
based on the `TROPICALNN_MODE` environment variable (`"highs"` by default,
or `"oscar"`).
"""
function lp_mode()
    mode_name = lowercase(get(ENV, "TROPICALNN_MODE", "highs"))
    if mode_name == "highs"
        return HiGHSMode()
    elseif mode_name == "oscar"
        return OscarMode()
    else
        error("Unknown TROPICALNN_MODE=\"$mode_name\"; expected \"highs\" or \"oscar\"")
    end
end

function plot_linear_regions(linear_regions; xlims=(-10.0, 10.0), ylims=(-10.0, 10.0), kwargs...)
    A_box = Rational{BigInt}[1 0; -1 0; 0 1; 0 -1]
    b_box = Rational{BigInt}.([xlims[2], -xlims[1], ylims[2], -ylims[1]])
    bbox = Oscar.polyhedron(A_box, b_box)
    A_box_float = Float64.(A_box)
    b_box_float = Float64.(b_box)

    p = plot(;
        xlim = xlims,
        ylim = ylims,
        legend = false,
        aspect_ratio = :equal,

        xticks = false,
        yticks = false,
        margin = 0mm,

        dpi = 300,

        kwargs...
    )
    cols = theme_palette(:auto)

    # for each region, dispatch to the HiGHS or Oscar plotting routine depending on
    # which polyhedron representation the region backend produced
    for (i, region) in enumerate(linear_regions)
        c = cols[mod1(i, length(cols))]

        polys = if region isa Tuple
            [region[1]]
        elseif !hasmethod(iterate, (typeof(region),))
            [region]
        else
            region
        end

        for poly in polys
            if hasproperty(poly, :A) && hasproperty(poly, :b)
                _plot_highs_region!(p, poly, A_box_float, b_box_float, c)
            else
                _plot_oscar_region!(p, poly, bbox, c)
            end
        end
    end
    
    return p
end
