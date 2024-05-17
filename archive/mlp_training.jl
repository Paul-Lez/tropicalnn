using Flux
using Distributions

# Simple multi-layer perceptron, for the MNIST hand-written digits.
# This example does not use a GPU, it's small enough not to need one.

using Flux, MLDatasets, Statistics

model = Chain(Dense(2 => 2),Dense(3=> 3, relu), Dense(3 => 2), softmax) #

#### in theory the network x => 

p1 = model(rand(Float32, 2))  # run model on random data 

@show sum(p1) ≈1;

p3 = model(rand(Float32, 2, 3))  # ...or on a batch of 3 fake data points

@show sum(p3; dims=1);  # all approx 1. Last dim is batch dim.

#===== DATA =====#

n_data = 5000
n_test = 500
train_data = rand(Normal(0, 1), 2, n_data)
test_data = rand(Normal(0, 1), 2, n_test)
train_output = [train_data[1, i] > 0 ? [1, 0] : [0, 1] for i in 1:n_data]
test_output = [test_data[1, i] > 0 ? [1, 0] : [0, 1] for i in 1:n_test]

#for i in eachindex(train_data)
#    println("x = ", train_data[1, i], " label = ", train_output[i])
#end 

# train_data.features is a 28×28×60000 Array{Float32, 3} of the images.
# We need a 2D array for our model. Let's combine the reshape needed with
# other pre-processing, in a function:

function simple_loader(data, batchsize::Int=64)
    train_data, train_output = data
    y = mapreduce(permutedims, vcat, train_output)
    ##yhot = Flux.onehotbatch(data.targets, 0:9)
    y =  reshape(y, 2, length(train_output))
    Flux.DataLoader((train_data, y); batchsize, shuffle=true)
end

model(train_data[:, 1])

test = rand(2)

@show size(model(test))


# train_data.targets is a 60000-element Vector{Int}, of labels from 0 to 9.
# Flux.onehotbatch([0,1,9], 0:9) makes a matrix of 0 and 1.

simple_loader((train_data, train_output))  # returns a DataLoader, with first element a tuple like this:

x1, y1 = first(simple_loader((train_data, train_output))); # (784×64 Matrix{Float32}, 10×64 OneHotMatrix)

@show size(model(x1))  # x1 is the right shape for our model

@show y1  # y1 is the same shape as the model output.

@show Flux.crossentropy(model(x1), y1);  # This will be our loss function

#===== ACCURACY =====#

# We're going to log accuracy and loss during training. There's no advantage to
# calculating these on minibatches, since MNIST is small enough to do it at once.

function simple_accuracy(train_data, train_output)
    (x, y) = only(simple_loader((train_data, train_output), n_data))  # make one big batch
    y_hat = model(x)
    iscorrect = Flux.onecold(y_hat) .== Flux.onecold(y)  # BitVector
    acc = round(100 * mean(iscorrect); digits=2)
    return acc
end

@show simple_accuracy(train_data, train_output);  

#===== TRAINING =====#

# Make a dataloader using the desired batchsize:

train_loader = simple_loader((train_data, train_output), 256)

# Initialise storage needed for the Adam optimiser, with our chosen learning rate:

opt_state = Flux.setup(Adam(3e-4), model);

# Then train for 30 epochs, printing out details as we go:

for epoch in 1:30
    loss = 0.0
    for (x, y) in train_loader
        # Compute the loss and the gradients:
        l, gs = Flux.withgradient(m -> Flux.crossentropy(m(x), y), model)
        # Update the model parameters (and the Adam momenta):
        Flux.update!(opt_state, model, gs[1])
        # Accumulate the mean loss, just for logging:
        loss += l / length(train_loader)
    end

    if mod(epoch, 2) == 1
        # Report on train and test, only every 2nd epoch:
        train_acc = simple_accuracy(train_data, train_output)
        test_acc = simple_accuracy(test_data, test_output)
        @info "After epoch = $epoch" loss train_acc test_acc
    end
end