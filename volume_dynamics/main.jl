using Flux
using Graphs
using JLD2
using TropicalNN
using Logging

global_logger(SimpleLogger(stderr, Logging.Error))

function f1(x,y)
    return [2*x-1, y+1.1]
end

function f2(x,y)
    return [2*x-1, -y+1.1]
end

function generate_data(num_train_per_cluster=48, num_test_per_cluster=16)

    X_train = Array{Float32}(undef, num_train_per_cluster*2, 2)
    Y_train = Int[]

    # Fix: Transpose f1/f2 outputs into row vectors first, then vcat
    X_train[1:num_train_per_cluster,:] =
        reduce(vcat, [f1(rand(),rand())' for _ in 1:num_train_per_cluster])

    X_train[num_train_per_cluster+1:end,:] =
        reduce(vcat, [f2(rand(),rand())' for _ in 1:num_train_per_cluster])

    append!(Y_train, fill(0, num_train_per_cluster))
    append!(Y_train, fill(1, num_train_per_cluster))

    X_test = Array{Float32}(undef, num_test_per_cluster*2, 2)
    Y_test = Int[]

    X_test[1:num_test_per_cluster,:] =
        reduce(vcat, [f1(rand(),rand())' for _ in 1:num_test_per_cluster])

    X_test[num_test_per_cluster+1:end,:] =
        reduce(vcat, [f2(rand(),rand())' for _ in 1:num_test_per_cluster])

    append!(Y_test, fill(0, num_test_per_cluster))
    append!(Y_test, fill(1, num_test_per_cluster))

    return X_train,X_test,Y_train,Y_test
end

# -------------------------------
# Accuracy
# -------------------------------

function get_accuracy(model,X,Y)
    X_mat = Matrix(X') 
    
    preds = model(X_mat) .>= 0.5
    correct = sum(preds .== reshape(Y,1,:))
    return correct / length(Y)
end

# -------------------------------
# Save weights
# -------------------------------

function save_weights(model,path)

    w = [Rational{BigInt}.(model[i].weight) for i in 1:length(model)-1]
    b = [Rational{BigInt}.(model[i].bias)   for i in 1:length(model)-1]

    mkpath(path)

    JLD2.save("$path/weights.jld2","data",w)
    JLD2.save("$path/biases.jld2","data",b)
end

# -------------------------------
# Training
# -------------------------------

function train(data_path,X_train,X_test,Y_train,Y_test;
               width=10,batch_size=16,lr=1e-3,max_epochs=200)

    w,b,t = random_mlp([2,width,1])

    model = Chain(
        Dense(w[1],b[1],relu),
        Dense(w[2],b[2],identity),
        σ
    )

    X_train_mat = Matrix(X_train')         
    Y_train_mat = reshape(Y_train, 1, :)   

    loader = Flux.DataLoader(
        (X_train_mat, Y_train_mat),
        batchsize=batch_size,
        shuffle=true
    )

    # 1. Update loss function to explicitly take the model 'm'
    loss(m, x, y) = Flux.binarycrossentropy(m(x), y)
    
    # 2. Set up the optimizer state explicitly
    opt_state = Flux.setup(Flux.Adam(lr), model)

    training_data = Dict(
        "loss"=>Float64[],
        "train_acc"=>Float64[],
        "test_acc"=>Float64[]
    )

    for epoch in 0:max_epochs

        epoch_loss = 0.0
        count = 0

        for (x_batch,y_batch) in loader
            
            # 3. Calculate both loss and gradients simultaneously!
            l, gs = Flux.withgradient(m -> loss(m, x_batch, y_batch), model)
            
            batch_n = size(y_batch, 2)
            epoch_loss += l * batch_n
            
            # 4. Update the model using the explicit state and gradients (gs[1] corresponds to the model)
            Flux.update!(opt_state, model, gs[1])
            
            count += batch_n
        end

        epoch_loss /= count

        train_acc = get_accuracy(model,X_train,Y_train)
        test_acc  = get_accuracy(model,X_test,Y_test)

        push!(training_data["loss"],epoch_loss)
        push!(training_data["train_acc"],train_acc)
        push!(training_data["test_acc"],test_acc)

        println("epoch $epoch  loss=$(round(epoch_loss,digits=4))  train=$(round(train_acc,digits=4))  test=$(round(test_acc,digits=4))")

        save_weights(model,"$data_path/$epoch")
    end

    return training_data
end

# -------------------------------
# Tropical analysis
# -------------------------------

function analyze_epochs(data_path)

    epochs = sort(parse.(Int, filter(x->!occursin(".",x), readdir(data_path))))

    monomial_data = Dict("pre"=>[],"post"=>[])

    for epoch in epochs

        println("Analyzing epoch $epoch")

        weights = JLD2.load("$data_path/$epoch/weights.jld2")["data"]
        biases  = JLD2.load("$data_path/$epoch/biases.jld2")["data"]

        t = [Rational{BigInt}.(zeros(length(b))) for b in biases]

        f_pre  = mlp_to_trop(weights,biases,t)[1]
        f_post = monomial_strong_elim(f_pre)

        G = get_graph(f_post)

        edge_data = Dict(
            "gradients" => edge_directions(G)["full"],
            "lengths"   => edge_lengths(G)["full"]
        )

        JLD2.save("$data_path/$epoch/graph.jld2", "graph", G)
        JLD2.save("$data_path/$epoch/edge_data.jld2", "data", edge_data)

        push!(monomial_data["pre"],  monomial_count(f_pre))
        push!(monomial_data["post"], monomial_count(f_post))
    end

    JLD2.save("$data_path/monomial_data.jld2","data",monomial_data)
end

# -------------------------------
# Main experiment
# -------------------------------

function run_experiment()

    data_path = "outputs/volume_dynamics"
    mkpath(data_path)

    X_train,X_test,Y_train,Y_test = generate_data()

    JLD2.save("$data_path/X_train.jld2","data",X_train)
    JLD2.save("$data_path/X_test.jld2","data",X_test)
    JLD2.save("$data_path/Y_train.jld2","data",Y_train)
    JLD2.save("$data_path/Y_test.jld2","data",Y_test)

    training_data = train(data_path,X_train,X_test,Y_train,Y_test)

    JLD2.save("$data_path/training_data.jld2","data",training_data)

    analyze_epochs(data_path)
end

run_experiment()