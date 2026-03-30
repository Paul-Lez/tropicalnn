using TropicalNN, Plots
include("../utils.jl")

output_dir = "../outputs/effective_radius/"
mkpath(output_dir)

w, b, t = random_mlp([2, 4, 1])
rmap = mlp_to_trop(w, b, t)[1]
er = exact_er(rmap)
linear_regions = enum_linear_regions_rat(rmap)

margin_limit = Float64(er * 1.2)
println(margin_limit)
fig = plot_linear_regions(linear_regions, xlims=(-margin_limit, margin_limit), ylims=(-margin_limit, margin_limit))

sq_x = [er, -er, -er,  er, er]
sq_y = [er,  er, -er, -er, er]

plot!(fig, sq_x, sq_y, 
      color=:red, 
      linewidth=2.5, 
      linestyle=:dash, 
      label="Effective Radius Square")

savefig(fig, joinpath(output_dir, "bounding_linear_regions.png"))