# Included for the `--width` / `MNIST_WIDTH` argument parsing helpers only; this
# script does not need a worker pool, so `setup_experiment!` is never called.
include(joinpath(@__DIR__, "..", "experiment_setup.jl"))

using Flux
using Flux: DataLoader, crossentropy, onecold, onehotbatch, setup, update!
using MLDatasets
using JLD2
using Statistics
using Printf

# Hidden width of the network. Keep `main.jl` and `analyse.jl` in sync by passing
# the same `--width` (or `MNIST_WIDTH`) to both.
const WIDTH = _parse_int(ARGS, ["--width"], "MNIST_WIDTH", 4)

# ==========================================
# 1. Load and Prepare Data
# ==========================================
function get_data(batch_size)
    # Pre-v0.7 MLDatasets syntax calls the module's functions directly
    X_train, y_train = MLDatasets.MNIST.traindata(Float32)
    X_test, y_test = MLDatasets.MNIST.testdata(Float32)

    # Reshape features to (features, N) and one-hot encode the digit labels.
    X_train = reshape(Float32.(X_train), 28^2, :)
    y_train = onehotbatch(y_train, 0:9)
    X_test = reshape(Float32.(X_test), 28^2, :)
    y_test = onehotbatch(y_test, 0:9)

    # Create DataLoaders to yield minibatches
    train_loader = DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)
    test_loader = DataLoader((X_test, y_test), batchsize=batch_size, shuffle=false)

    return train_loader, test_loader, (X_train, y_train), (X_test, y_test)
end

# ==========================================
# 2. Define the Model
# ==========================================
function build_model(width)
    Chain(
        Dense(28^2 => width, relu),
        Dense(width => 10),
        softmax
    )
end

# Helper function to calculate 10-class accuracy.
accuracy(model, x, y) = mean(onecold(model(x)) .== onecold(y))

# ==========================================
# 3. Main Training Routine
# ==========================================
function train_and_save()
    width = WIDTH
    epochs = 30
    batch_size = 128
    learning_rate = 0.005

    println("Loading data...")
    train_loader, test_loader, train_full, test_full = get_data(batch_size)

    println("Building MLP model with architecture [$(28^2), $width, 10]...")
    model = build_model(width)

    # Setup the Adam optimizer
    opt_state = Flux.setup(Adam(learning_rate), model)

    println("Starting training loop...")
    for epoch in 1:epochs
        loss_sum = 0.0
        batch_count = 0

        for (x, y) in train_loader
            # The model returns softmax probabilities, so use cross-entropy.
            loss_val, grads = Flux.withgradient(model) do m
                crossentropy(m(x), y)
            end

            # Update the model parameters
            Flux.update!(opt_state, model, grads[1])

            loss_sum += loss_val
            batch_count += 1
        end

        avg_loss = loss_sum / batch_count
        @printf("Epoch %d/%d - Average Loss: %.4f\n", epoch, epochs, avg_loss)
    end

    # Calculate final accuracies
    println("Calculating final accuracies...")
    train_acc = accuracy(model, train_full[1], train_full[2])
    test_acc = accuracy(model, test_full[1], test_full[2])

    @printf("Train Accuracy: %.2f%%\n", train_acc * 100)
    @printf("Test Accuracy: %.2f%%\n", test_acc * 100)

    # ==========================================
    # 4. Save the Model and Metrics
    # ==========================================
    println("Saving results to outputs/mnist/...")
    output_dir = "outputs/mnist"
    mkpath(output_dir)

    # Save only the model state (weights/biases), keyed by width so that models
    # of different widths do not overwrite each other.
    model_state = Flux.state(model)
    jldsave(joinpath(output_dir, "model_$width.jld2"); model_state)

    # Save the evaluation metrics
    open(joinpath(output_dir, "metrics_$width.txt"), "w") do io
        write(io, "Model Type: MLP (Architecture: [$(28^2), $width, 10])\n")
        write(io, "Train Accuracy: $(round(train_acc * 100, digits=2))%\n")
        write(io, "Test Accuracy: $(round(test_acc * 100, digits=2))%\n")
    end

    println("Done! Model state saved successfully.")
end

# Run the script
train_and_save()
