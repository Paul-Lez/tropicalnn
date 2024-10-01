include("../src/rat_maps.jl")
include("../src/linear_regions.jl")
include("../src/mlp_to_trop.jl")
include("../src/mlp_to_trop_with_elim.jl")

using Flux
using BSON: @load
using JLD2
using Oscar

# set the width of the hidden layer of the neural network to be equal to that of the trained model
width=4

# function to extract the parameters required to compute the tropical representation
function extract_weights_biases_thresholds(model,symbolic=true)
    if symbolic
        weights = [Rational{BigInt}.(model[i].weight) for i in 1:length(model)-1]
        biases = [Rational{BigInt}.(model[i].bias) for i in 1:length(model)-1]
        thresholds = [Rational{BigInt}.(zeros(length(model[i].bias))) for i in 1:length(model)-1]
    else
        weights = [model[i].weight for i in 1:length(model)-1]
        biases = [model[i].bias for i in 1:length(model)-1]
        thresholds = [zeros(length(model[i].bias)) for i in 1:length(model)-1]
    end
    return weights,biases,thresholds
end

# load in the pretrained model
model = Chain(Dense(28^2 => width, relu), Dense(width => 10), softmax)
@load "mnist/trained_model_$width.bson" model

# analyse the pre-trained model
start_time=time()
weights,biases,thresholds=extract_weights_biases_thresholds(model)
output=mlp_to_trop_with_quicksum(weights,biases,thresholds)
println("got tropical respresentation")
lin_regions=enum_linear_regions_rat(output[1].num,output[1].den)
println("Number of linear regions is "*string(length(lin_regions)))
mon_count=monomial_count(output)
println("Monomial count is $mon_count")
analysis_time=time()-start_time
println("time $analysis_time")

analysis=Dict()
analysis["trop_rep"]=output
analysis["num_lin_region"]=length(lin_regions)
analysis["num_mon"]=mon_count
analysis["time"]=analysis_time
JLD2.save("mnist/analysis_$width.jld2")