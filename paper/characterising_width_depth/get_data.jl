include("../src/rat_maps.jl")
include("../src/linear_regions.jl")
include("../src/mlp_to_trop.jl")
include("../src/mlp_to_trop_with_elim.jl")

using JLD2
using Oscar

# get the number of monomials in randomly initialised neural networks of a certain architecture
# strong_elim determines whether the tropical expression is pruned before counting its monomials
function get_data(widths,max_num_samples,strong_elim=false)
    input_dim=widths[1]
    if strong_elim
        if isfile("characterising_width_depth/data_strongelim_$input_dim.jld2")
            data=JLD2.load("characterising_width_depth/data_strongelim_$input_dim.jld2")["data"]
        else
            data=Dict()
        end
    else
        if isfile("characterising_width_depth/data_$input_dim.jld2")
            data=JLD2.load("characterising_width_depth/data_$input_dim.jld2")["data"]
        else
            data=Dict()
        end
    end

    if haskey(data,widths)
        num_samples=max(0,max_num_samples-length(data[widths]["monomial_count"]))
    else
        num_samples=max_num_samples
    end
    for i in 1:num_samples
        start_time=time()
        weights,biases,thresholds=random_mlp(widths)
        if strong_elim
            output=mlp_to_trop_with_quicksum_with_strong_elim(weights,biases,thresholds)
        else
            output=mlp_to_trop_with_quicksum(weights,biases,thresholds)
        end
        
        mon_count=monomial_count(output)
        total_time=time()-start_time

        if haskey(data,widths)
            push!(data[widths]["monomial_count"],mon_count)
            push!(data[widths]["time"],total_time)
        else
            data[widths]=Dict()
            data[widths]["monomial_count"]=[mon_count]
            data[widths]["time"]=[total_time]
        end
        println("Width $widths - $i/$num_samples")
        if strong_elim
            JLD2.save("characterising_width_depth/data_strongelim_$input_dim.jld2","data",data)
        else
            JLD2.save("characterising_width_depth/data_$input_dim.jld2","data",data)
        end
    end
end