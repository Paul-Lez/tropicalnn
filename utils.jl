using TropicalNN
using Plots
using Plots.PlotMeasures
using Oscar

function plot_linear_regions(lrs; xlims=(-10.0, 10.0), ylims=(-10.0, 10.0), kwargs...)
    A_box = [1.0  0.0; -1.0  0.0; 0.0  1.0; 0.0 -1.0]
    b_box = [xlims[2], -xlims[1], ylims[2], -ylims[1]]
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

    for (i, item) in enumerate(lrs)
        c = cols[mod1(i, length(cols))]
        
        polys = if item isa Tuple
            [item[1]]
        elseif !hasmethod(iterate, (typeof(item),))
            [item]
        else
            item
        end
        
        for poly in polys
            bounded_poly = Oscar.intersect(poly, bbox)
            
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