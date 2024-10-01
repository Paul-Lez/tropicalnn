include("../src/rat_maps.jl")
include("../src/linear_regions.jl")
include("../src/mlp_to_trop.jl")
include("../src/mlp_to_trop_with_elim.jl")

using JLD2
using Oscar
using Flux

# calculates the rate of pruning for randomly initialised networks
function get_data_untrained(network_width,network_layers,max_num_samples)
    if isfile("redundant_monomials/rm_$network_layers.jld2")
        data=JLD2.load("redundant_monomials/rm_$network_layers.jld2")["data"]
    else
        data=Dict()
    end
    if haskey(data,network_width)
        num_samples=max(0,max_num_samples-length(data[network_width]["native"]))
    else
        num_samples=max_num_samples
    end
    for i in 1:num_samples
        widths=[2]
        widths=vcat(widths,repeat([network_width],network_layers))
        widths=vcat(widths,[1])
        start_time=time()
        weights,biases,thresholds=random_mlp(widths)
        output_native=mlp_to_trop_with_quicksum(weights,biases,thresholds)
        output_reduced=mlp_to_trop_with_quicksum_with_strong_elim(weights,biases,thresholds)
        mon_count_native=monomial_count(output_native)
        mon_count_reduced=monomial_count(output_reduced)
        total_time=time()-start_time

        if haskey(data,network_width)
            push!(data[network_width]["native"],mon_count_native)
            push!(data[network_width]["reduced"],mon_count_reduced)
            push!(data[network_width]["time"],total_time)
        else
            data[network_width]=Dict()
            data[network_width]["native"]=[mon_count_native]
            data[network_width]["reduced"]=[mon_count_reduced]
            data[network_width]["time"]=[total_time]
        end
        println("Untrained - width $widths - $i/$num_samples")
        JLD2.save("red_mon/rm_$network_layers.jld2","data",data)
    end
end

# calculates the rate of pruning for trained networks
function get_data_trained(network_width,network_layers,max_num_samples,num_samples_per_cluster=64,epochs=20,batch_size=8,lr=0.01,spread=0.1)
    if isfile("redundant_monomials/rm_trained_$network_layers.jld2")
        data=JLD2.load("redundant_monomials/rm_trained_$network_layers.jld2")["data"]
    else
        data=Dict()
    end
    if haskey(data,network_width)
        num_samples=max(0,max_num_samples-length(data[network_width]["native"]))
    else
        num_samples=max_num_samples
    end
    for i in 1:num_samples
        widths=[2]
        widths=vcat(widths,repeat([network_width],network_layers))
        widths=vcat(widths,[1])

        start_time=time()
        # create model
        layers = []
        push!(layers, Dense(widths[1], widths[2], relu))
        if length(widths)==3
            push!(layers, Dense(widths[2], widths[3]))
        else
            for i in 3:length(widths)-1
                push!(layers, Dense(widths[i-1], widths[i],relu))
            end
            push!(layers, Dense(widths[length(widths)-1], widths[end]))
        end
        model=Chain(layers...,σ)
        
        #generate data
        X=Array{Float32}(undef, num_samples_per_cluster * 2, widths[1])
        Y=[]
    
        for num_cluster in 1:2
            center=randn(1,widths[1])*2
            center[1]=0.5*cos((num_cluster-1)*2*3.14/2)
            center[2]=0.5*sin((num_cluster-1)*2*3.14/2)
            points_cluster=randn(num_samples_per_cluster,2)*spread.+center
            X[num_samples_per_cluster*(num_cluster-1)+1:num_samples_per_cluster*num_cluster,:]=points_cluster
            append!(Y,fill(num_cluster-1,num_samples_per_cluster))
        end
        # train
        loader = Flux.Data.DataLoader((data=X',label=Y), batchsize=batch_size, shuffle=true)
        loss(x, y) = Flux.binarycrossentropy(model(x), reshape(y, 1, :))
        opt = Flux.Adam(lr)
        for epoch in 1:epochs
            epoch_loss=0
            for (x_batch, y_batch) in loader
                gs = Flux.gradient(Flux.params(model)) do
                    l = loss(x_batch, y_batch)
                    epoch_loss += l
                    return l
                end
                
                Flux.Optimise.update!(opt, Flux.params(model), gs)
            end
        end
        # extrace weights and biases
        weights = [Rational{BigInt}.(model[i].weight) for i in 1:length(model)-1]
        biases = [Rational{BigInt}.(model[i].bias) for i in 1:length(model)-1]
        thresholds = [Rational{BigInt}.(zeros(length(model[i].bias))) for i in 1:length(model)-1]

        preds = model(X') .>= 0.5
        correct = sum(preds .== reshape(Y, 1, :))
        acc=correct/length(Y)
        println(acc)

        output_native=mlp_to_trop_with_quicksum(weights,biases,thresholds)
        output_reduced=mlp_to_trop_with_quicksum_with_strong_elim(weights,biases,thresholds)
        mon_count_native=monomial_count(output_native)
        mon_count_reduced=monomial_count(output_reduced)
        total_time=time()-start_time
        if haskey(data,network_width)
            push!(data[network_width]["native"],mon_count_native)
            push!(data[network_width]["reduced"],mon_count_reduced)
            push!(data[network_width]["accuracy"],acc)
            push!(data[network_width]["time"],total_time)
        else
            data[network_width]=Dict()
            data[network_width]["native"]=[mon_count_native]
            data[network_width]["reduced"]=[mon_count_reduced]
            data[network_width]["accuracy"]=[acc]
            data[network_width]["time"]=[total_time]
        end
        println("Trained - width $widths - $i/$num_samples")
    end
    JLD2.save("redundant_monomials/rm_trained_$network_layers.jld2","data",data)
end