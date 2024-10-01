using JLD2

# summarises the collected data
function summarise_data()
    data=JLD2.load("num_linear_regions/region_counts.jld2")["data"]
    total_time=0
    for (key,values) in data
        avg_num_lin_regions=sum(values["num_linear_regions"])/length(values["num_linear_regions"])
        max_num_lin_regions=maximum(values)
        avg_time=sum(values["time"])/length(values["time"])
        n_samples=length(values["num_linear_regions"])
        println("Architecture $key - $avg_num_lin_regions avg num linear regions - $avg_time average time - $n_samples samples")
        total_time+=sum(values["time"])
    end
    println("Total time: $total_time")
end