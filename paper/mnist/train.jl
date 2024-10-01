using Flux 
using MLDatasets
using Statistics

using BSON: @save
using JLD2

# set the width of the hidden layer of the neural network
width=4
model = Chain(Dense(28^2 => width, relu), Dense(width => 10), softmax)

# import the test and train data
train_data = MLDatasets.MNIST()
test_data = MLDatasets.MNIST(split=:test)

function simple_loader(data::MNIST; batchsize::Int=64)
    x2dim = reshape(data.features, 28^2, :)
    yhot = Flux.onehotbatch(data.targets, 0:9)
    Flux.DataLoader((x2dim, yhot); batchsize, shuffle=true)
end

function simple_accuracy(model, data::MNIST=test_data)
    (x, y) = only(simple_loader(data; batchsize=length(data)))
    y_hat = model(x)
    iscorrect = Flux.onecold(y_hat) .== Flux.onecold(y)
    acc = round(100 * mean(iscorrect); digits=2)
end

# train the neural network
train_loader = simple_loader(train_data, batchsize = 256)
opt_state = Flux.setup(Adam(3e-4), model);

for epoch in 1:30
    loss = 0.0
    for (x, y) in train_loader
        l, gs = Flux.withgradient(m -> Flux.crossentropy(m(x), y), model)
        Flux.update!(opt_state, model, gs[1])
        loss += l / length(train_loader)
    end

    if mod(epoch, 2) == 1
        train_acc = simple_accuracy(model, train_data)
        test_acc = simple_accuracy(model, test_data)
        @info "After epoch = $epoch" loss train_acc test_acc
    end
end

# save the trained model for later analysis
@save "mnist/trained_model_$width.bson" model

# store the accuracy of the model
accuracy=Dict()
accuracy["train"]=simple_accuracy(model, train_data)
accuracy["test"]=simple_accuracy(model, test_data)
JLD2.save("mnist/accuracy_$width.jld2",accuracy)