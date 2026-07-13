using Flux
using Flux: DataLoader, setup, update!
using MLDatasets
using JLD2
using Statistics
using Printf

# ==========================================
# 1. Load and Prepare Data
# ==========================================
function get_data(batch_size)
    # Pre-v0.7 MLDatasets syntax calls the module's functions directly
    X_train, y_train_digits = MLDatasets.MNIST.traindata(Float32)
    X_test, y_test_digits = MLDatasets.MNIST.testdata(Float32)

    # Reshape features to (features, N)
    X_train = reshape(Float32.(X_train), 28^2, :)
    X_test = reshape(Float32.(X_test), 28^2, :)

    # Binary label: 1 if the digit shown is a 0, else 0. Kept as a (1, N)
    # matrix so it lines up with the model's scalar output.
    y_train = reshape(Float32.(y_train_digits .== 0), 1, :)
    y_test = reshape(Float32.(y_test_digits .== 0), 1, :)

    # Create DataLoaders to yield minibatches
    train_loader = DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)
    test_loader = DataLoader((X_test, y_test), batchsize=batch_size, shuffle=false)

    return train_loader, test_loader, (X_train, y_train), (X_test, y_test)
end

# ==========================================
# 2. Define the Model
# ==========================================
function build_model(width, width2)
    Chain(
        Dense(28^2 => width, relu),
        Dense(width => width2, relu),
        Dense(width2 => 1),
        # binary out: "is the digit 0?" classification
        sigmoid
    )
end

# Helper function to calculate accuracy for the binary (0 vs not-0) task
accuracy(model, x, y) = mean((model(x) .> 0.5f0) .== (y .> 0.5f0))

# ==========================================
# 3. Main Training Routine
# ==========================================
function train_and_save()
    width = 5
    width2 = 5
    epochs = 30
    batch_size = 128
    learning_rate = 0.005

    println("Loading data...")
    train_loader, test_loader, train_full, test_full = get_data(batch_size)

    println("Building MLP model with widths = $width, $width2...")
    model = build_model(width, width2)

    # Setup the Adam optimizer
    opt_state = Flux.setup(Adam(learning_rate), model)

    println("Starting training loop...")
    for epoch in 1:epochs
        loss_sum = 0.0
        batch_count = 0

        for (x, y) in train_loader
            # Because the model ends in `sigmoid`, we use standard
            # `binarycrossentropy` instead of `logitbinarycrossentropy`
            loss_val, grads = Flux.withgradient(model) do m
                mean(Flux.binarycrossentropy(m(x), y))
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

    # Save only the model state (weights/biases)
    model_state = Flux.state(model)
    jldsave(joinpath(output_dir, "model.jld2"); model_state)

    # Save the evaluation metrics
    open(joinpath(output_dir, "metrics.txt"), "w") do io
        write(io, "Model Type: MLP (Widths: $width, $width2), binary output (digit == 0)\n")
        write(io, "Train Accuracy: $(round(train_acc * 100, digits=2))%\n")
        write(io, "Test Accuracy: $(round(test_acc * 100, digits=2))%\n")
    end

    println("Done! Model state saved successfully.")
end

# Run the script
train_and_save()
