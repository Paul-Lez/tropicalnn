include(joinpath(@__DIR__, "..", "experiment_setup.jl"))
const EXPERIMENT_RUNTIME = setup_experiment!()

using TropicalNN
import Flux
using JLD2

const REGION_MODE = highs_mode(EXPERIMENT_RUNTIME)
const WORKER_IDS = tropical_workers(EXPERIMENT_RUNTIME)

# Set the hidden width of the neural network (must match main.jl).
width = _parse_int(ARGS, ["--width"], "MNIST_WIDTH", 4)
println("Analysing the MNIST network of hidden width $width")

# Function to extract the parameters required to compute the tropical representation
function extract_weights_biases_thresholds(model, symbolic=true)
    # Skip the final softmax layer because it has no learnable parameters and
    # is not piecewise linear.
    num_dense_layers = length(model) - 1 
    
    if symbolic
        weights = [Rational{BigInt}.(model[i].weight) for i in 1:num_dense_layers]
        biases = [Rational{BigInt}.(model[i].bias) for i in 1:num_dense_layers]
        thresholds = [Rational{BigInt}.(zeros(length(model[i].bias))) for i in 1:(num_dense_layers-1)]
    else
        weights = [Float64.(model[i].weight) for i in 1:num_dense_layers]
        biases = [Float64.(model[i].bias) for i in 1:num_dense_layers]
        thresholds = [zeros(Float64, length(model[i].bias)) for i in 1:(num_dense_layers-1)]
    end
    
    return weights, biases, thresholds
end

# 1. Define the exact same architecture that was trained
model = Flux.Chain(
    Flux.Dense(28^2 => width, Flux.relu),
    Flux.Dense(width => 10),
    Flux.softmax,
)

# 2. Load the state into the model
println("Loading trained model...")
# We use JLD2.load to grab the state dictionary we saved earlier, then apply it
model_state = JLD2.load("outputs/mnist/model_$width.jld2", "model_state")
Flux.loadmodel!(model, model_state)

# 3. Analyze the pre-trained model
println("Extracting weights and biases...")
start_time = time()

weights, biases, thresholds = extract_weights_biases_thresholds(model, false)

println("Computing tropical representation...")
tropicalize_time = @elapsed output = TropicalNN.tropicalize(weights, biases, thresholds,
    quicksum=true, prune=true, dedup=true, elim_mode=REGION_MODE, workers=WORKER_IDS)
println("Got tropical representation! ($tropicalize_time seconds)")

println("Enumerating linear regions...")
# Pass the whole output vector so we get the linear regions of the full network
# (the common refinement over all 10 output coordinates), not just of output 1.
regions_time = @elapsed lin_regions = TropicalNN.linear_regions(output;
    mode=REGION_MODE, workers=WORKER_IDS)
println("Number of linear regions is: ", length(lin_regions),
    " ($regions_time seconds)")

println("Counting monomials...")
mon_count = TropicalNN.monomial_count(output)
println("Monomial count is: $mon_count")

analysis_time = time() - start_time
println("Total analysis time: $analysis_time seconds")

# 4. Save the analysis correctly
analysis = Dict(
    "trop_rep" => output,
    "num_lin_region" => length(lin_regions),
    "num_mon" => mon_count,
    "time" => analysis_time,
    "tropicalize_time" => tropicalize_time,
    "linear_regions_time" => regions_time,
    "width" => width
)

println("Saving analysis...")
# Make sure the directory exists before attempting to save
output_dir = "outputs/mnist"
mkpath(output_dir)

# Using jldsave to explicitly save the dictionary object into the file
jldsave(joinpath(output_dir, "analysis_$width.jld2"); analysis)

println("Done!")
