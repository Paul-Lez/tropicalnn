using TropicalNN
using Plots
using Plots.PlotMeasures
using Oscar

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

"""
    _as_oscar_polyhedron(poly; mode)

Return an exact `Oscar.Polyhedron` for a region produced by `linear_regions`,
regardless of which `LinearRegionsCalculationMode` backend was used.
"""
function _as_oscar_polyhedron(poly; mode)
    if mode isa OscarMode
        return poly
    end
    A = get_matrix(poly; mode = mode)
    b = get_vector(poly; mode = mode)
    A_rat = Rational{BigInt}[rationalize(BigInt, x; tol = 1e-8) for x in A]
    b_rat = Rational{BigInt}[rationalize(BigInt, x; tol = 1e-8) for x in b]
    return Oscar.polyhedron(A_rat, b_rat)
end

function plot_linear_regions(linear_regions; mode = lp_mode(), xlims=(-10.0, 10.0), ylims=(-10.0, 10.0), kwargs...)
    A_box = Rational{BigInt}[1 0; -1 0; 0 1; 0 -1]
    b_box = Rational{BigInt}.([xlims[2], -xlims[1], ylims[2], -ylims[1]])
    bbox = Oscar.polyhedron(A_box, b_box)

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

    # for each region, convert to an Oscar polyhedron, intersect with the bounding box,
    # and plot the resulting polygon if it is feasible and full-dimensional
    for (i, region) in enumerate(linear_regions)
        c = cols[mod1(i, length(cols))]

        for poly in region
            oscar_poly = _as_oscar_polyhedron(poly; mode = mode)
            bounded_poly = Oscar.intersect(oscar_poly, bbox)

            if Oscar.is_feasible(bounded_poly) && Oscar.is_fulldimensional(bounded_poly)
                verts = collect(Oscar.vertices(bounded_poly))
                if isempty(verts)
                    continue
                end
                
                x_coords = [Float64(v[1]) for v in verts]
                y_coords = [Float64(v[2]) for v in verts]
                
                cx = sum(x_coords) / length(x_coords)
                cy = sum(y_coords) / length(y_coords)
                angles = atan.(y_coords .- cy, x_coords .- cx)
                perm = sortperm(angles)
                
                plot!(p, x_coords[perm], y_coords[perm], seriestype=:shape, color=c, alpha=0.6, linecolor=:black, linewidth=1)
            end
        end
    end
    
    return p
end
