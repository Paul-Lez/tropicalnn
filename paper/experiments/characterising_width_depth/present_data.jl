using JLD2
using Plots

function get_time(input_dims,strong_elim=false)
    total_time=0
    for input_dim in input_dims
        dim_time=0
        if strong_elim
            data=JLD2.load("characterising_width_depth/data_strongelim_$input_dim.jld2")["data"]
        else
            data=JLD2.load("characterising_width_depth/data_$input_dim.jld2")["data"]
        end
        for (key,value) in data
            value=value["time"]
            dim_time+=sum(value)
            samples=length(value)
            println("$input_dim $key: $samples")
        end
        println("Input dimension $input_dim: $dim_time")
        total_time+=dim_time
    end
    println("Total: $total_time")
end

function get_plots(input_dims,strong_elim=false)
    scaling_one_layer=Dict()
    scaling_two_layer=Dict()

    for input_dim in input_dims
        if strong_elim
            data=JLD2.load("characterising_width_depth/data_strongelim_$input_dim.jld2")["data"]
        else
            data=JLD2.load("characterising_width_depth/data_$input_dim.jld2")["data"]
        end

        for (key,value) in data
            value=value["monomial_count"]
            if length(key)==3
                if haskey(scaling_one_layer,key[1])
                    scaling_one_layer[key[1]][key[2]]=sum(value)/length(value)
                else
                    scaling_one_layer[key[1]]=Dict()
                    scaling_one_layer[key[1]][key[2]]=sum(value)/length(value)
                end
            end
            if length(key)==4 && key[2]==2
                if haskey(scaling_two_layer,key[1])
                    scaling_two_layer[key[1]][key[3]]=sum(value)/length(value)
                else
                    scaling_two_layer[key[1]]=Dict()
                    scaling_two_layer[key[1]][key[3]]=sum(value)/length(value)
                end
            end
        end
    end
    colors=Plots.cgrad(:Paired_8,length(collect(keys(scaling_one_layer))),categorical=true)
    plot(legend=:topleft,xlabel="Width",ylabel="Number of Monomials")
    for k in 1:length(collect(keys(scaling_one_layer)))
        input_dim=sort(collect(keys(scaling_one_layer)))[k]
        color=colors[k]

        Plots.plot!(sort(collect(keys(scaling_one_layer[input_dim]))),[scaling_one_layer[input_dim][key] for key in sort(collect(keys(scaling_one_layer[input_dim])))],color=color,label=string(input_dim),lw=2)
        Plots.plot!(sort(collect(keys(scaling_two_layer[input_dim]))),[scaling_two_layer[input_dim][key] for key in sort(collect(keys(scaling_two_layer[input_dim])))],line=(2,:dash),color=color,label="")
        if strong_elim
            savefig("characterising_width_depth/depth_strongelim_analysis.png")
        else
            savefig("characterising_width_depth/depth_analysis.png")
        end
    end
    plot(legend=:topleft,xlabel="Width",ylabel="Number of Monomials")
    for k in 1:length(collect(keys(scaling_one_layer)))
        input_dim=sort(collect(keys(scaling_one_layer)))[k]
        color=colors[k]

        Plots.plot!(sort(collect(keys(scaling_one_layer[input_dim]))),[scaling_one_layer[input_dim][key] for key in sort(collect(keys(scaling_one_layer[input_dim])))],color=color,yscale=:log2,label=string(input_dim),lw=2)
        Plots.plot!(sort(collect(keys(scaling_two_layer[input_dim]))),[scaling_two_layer[input_dim][key] for key in sort(collect(keys(scaling_two_layer[input_dim])))],line=(2,:dash),color=color,yscale=:log2,label="")
        if strong_elim
            savefig("characterising_width_depth/depth_strongelim_analysis_log.png")
        else
            savefig("characterising_width_depth/depth_analysis_log.png")
        end
    end
end