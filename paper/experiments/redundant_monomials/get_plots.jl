using CairoMakie
using JLD2

function layer_plot(layers)
    data=JLD2.load("red_mon/rm_$layers.jld2")
    sorted_keys=sort(collect(keys(data["data"])))
    sorted_values=[data["data"][key] for key in sorted_keys]
    sampled_widths=[]
    ratios=[]
    total_time=0
    for k in 1:length(sorted_keys)
        println("Untrained - "*string(sorted_keys[k])*" "*string(length(sorted_values[k]["native"])))
        push!(sampled_widths,sorted_keys[k])
        native_counts_avg=sum(sorted_values[k]["native"])/length(sorted_values[k]["native"])
        reduced_counts_avg=sum(sorted_values[k]["reduced"])/length(sorted_values[k]["reduced"])
        push!(ratios,1-reduced_counts_avg/native_counts_avg)
        total_time+=sum(sorted_values[k]["time"])
    end
    f=Figure()
    ax=Axis(f[1,1],xlabel="Layer Width",ylabel="Pruning Rate",limits=(nothing,nothing,0,1.02))
    lines!(ax,sampled_widths,ratios,color=:blue,label="untrained")


    data=JLD2.load("red_mon/rm_trained_$layers.jld2")
    sorted_keys=sort(collect(keys(data["data"])))
    sorted_values=[data["data"][key] for key in sorted_keys]
    sampled_widths=[]
    ratios=[]
    accuracies=[]
    for k in 1:length(sorted_keys)
        println("Trained - "*string(sorted_keys[k])*" "*string(length(sorted_values[k]["native"])))
        push!(sampled_widths,sorted_keys[k])
        native_counts_avg=sum(sorted_values[k]["native"])/length(sorted_values[k]["native"])
        reduced_counts_avg=sum(sorted_values[k]["reduced"])/length(sorted_values[k]["reduced"])
        accuracy_avg=sum(sorted_values[k]["accuracy"])/length(sorted_values[k]["accuracy"])
        push!(ratios,1-reduced_counts_avg/native_counts_avg)
        push!(accuracies,accuracy_avg)
        total_time+=sum(sorted_values[k]["time"])
    end

    ax2=Axis(f[1,1],xlabel="Layer Width",ylabel="Accuracy",yticklabelcolor=:red,yaxisposition=:right,limits=(nothing,nothing,0,1.02))
    hidespines!(ax2)
    hidexdecorations!(ax2)
    lines!(ax,sampled_widths,ratios,color=:blue,linestyle=:dash,label="trained")
    lines!(ax2,sampled_widths,accuracies,color=:red,linestyle=:dash)
    axislegend(ax,halign=:left,valign=:center)
    save("red_mon/layer_$layers.png",f)
    
    println("time: $total_time")
end