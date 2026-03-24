using TropicalNN
using Flux
using JLD2

# Set the width of the hidden layer of the neural network
width = 4

# Function to extract the parameters required to compute the tropical representation
function extract_weights_biases_thresholds(model, symbolic=true)
    # The loop goes up to length(model)-1 to deliberately skip the final 'softmax' layer,
    # because softmax does not have learnable weights/biases and isn't piecewise linear.
    num_dense_layers = length(model) - 1 
    
    if symbolic
        weights = [Rational{BigInt}.(model[i].weight) for i in 1:num_dense_layers]
        biases = [Rational{BigInt}.(model[i].bias) for i in 1:num_dense_layers]
        thresholds = [Rational{BigInt}.(zeros(length(model[i].bias))) for i in 1:num_dense_layers]
    else
        weights = [model[i].weight for i in 1:num_dense_layers]
        biases = [model[i].bias for i in 1:num_dense_layers]
        thresholds = [zeros(length(model[i].bias)) for i in 1:num_dense_layers]
    end
    
    return weights, biases, thresholds
end

# 1. Define the exact same architecture that was trained
model = Chain(Dense(28^2 => width, relu), Dense(width => 10), softmax)

# 2. Load the state into the model (Fixing the path typo and loading mechanism)
println("Loading trained model...")
# We use JLD2.load to grab the state dictionary we saved earlier, then apply it
model_state = JLD2.load("outputs/mnist/model.jld2", "model_state")
Flux.loadmodel!(model, model_state)

# 3. Analyze the pre-trained model
println("Extracting weights and biases...")
start_time = time()
weights, biases, thresholds = extract_weights_biases_thresholds(model)

println("Computing tropical representation...")
output = TropicalNN.mlp_to_trop(weights, biases, thresholds, quicksum=true)
println("Got tropical representation!")

println("Enumerating linear regions...")
lin_regions = TropicalNN.enum_linear_regions_rat(output[1]) #_highs
println("Number of linear regions is: ", length(lin_regions))

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
    "time" => analysis_time
)

println("Saving analysis...")
# Make sure the directory exists before attempting to save
output_dir = "../outputs/mnist"
mkpath(output_dir)

# Using jldsave to explicitly save the dictionary object into the file
jldsave(joinpath(output_dir, "analysis_$width.jld2"); analysis)

println("Done!")