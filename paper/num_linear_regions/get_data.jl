include("../src/rat_maps.jl")
include("../src/linear_regions.jl")
include("../src/mlp_to_trop.jl")
include("../src/mlp_to_trop_with_elim.jl")

using JLD2
using Oscar
using Combinatorics

# Obtains the number of linear regions for randomly initialised neural networks of a particular architecture
# samples until the maximum number of required samples is reached is the data
function get_data(architecture,max_num_samples=25)
    if isfile("num_linear_regions/region_counts.jld2")
        data=JLD2.load("num_linear_regions/region_counts.jld2")["data"]
    else
        data=Dict()
    end
    if haskey(data,architecture)
        num_samples=max(0,max_num_samples-length(data[architecture]["num_linear_regions"]))
    else
        num_samples=max_num_samples
    end
    for i in 1:num_samples
        weights,biases,thresholds=random_mlp(architecture)
        start_time=time()
        trop=mlp_to_trop_with_quicksum(weights,biases,thresholds)[1]
        lin_regions=enum_linear_regions_rat(trop.num,trop.den)
        num_lin_regions=length(lin_regions)
        total_time=time()-start_time

        if haskey(data,architecture)
            push!(data[architecture]["num_linear_regions"],num_lin_regions)
            push!(data[architecture]["time"],total_time)
        else
            data[architecture]=Dict()
            data[architecture]["num_linear_regions"]=[num_lin_regions]
            data[architecture]["time"]=[total_time]
        end

        println("Architecture $architecture - $i/$num_samples")
        JLD2.save("num_linear_regions/region_counts.jld2","data",data)
    end
end