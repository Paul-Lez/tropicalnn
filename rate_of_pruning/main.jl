using Flux
using TropicalNN
using DataFrames
using CSV
using Logging

global_logger(SimpleLogger(stderr, Logging.Error))

function f1(x, y)
    return [2*x - 1, y + 1.1]
end

function f2(x, y)
    return [2*x - 1, -y + 1.1]
end

function generate_data(num_train_per_cluster=48, num_test_per_cluster=16)
    X_train = Array{Float32}(undef, num_train_per_cluster * 2, 2)
    Y_train = Int[]

    X_train[1:num_train_per_cluster, :] =
        reduce(vcat, [f1(rand(), rand())' for _ in 1:num_train_per_cluster])

    X_train[num_train_per_cluster+1:end, :] =
        reduce(vcat, [f2(rand(), rand())' for _ in 1:num_train_per_cluster])

    append!(Y_train, fill(0, num_train_per_cluster))
    append!(Y_train, fill(1, num_train_per_cluster))

    X_test = Array{Float32}(undef, num_test_per_cluster * 2, 2)
    Y_test = Int[]

    X_test[1:num_test_per_cluster, :] =
        reduce(vcat, [f1(rand(), rand())' for _ in 1:num_test_per_cluster])

    X_test[num_test_per_cluster+1:end, :] =
        reduce(vcat, [f2(rand(), rand())' for _ in 1:num_test_per_cluster])

    append!(Y_test, fill(0, num_test_per_cluster))
    append!(Y_test, fill(1, num_test_per_cluster))

    return X_train, X_test, Y_train, Y_test
end

function get_monomial_counts(model)
    w = [Rational{BigInt}.(model[i].weight) for i in 1:length(model)-1]
    b = [Rational{BigInt}.(model[i].bias)   for i in 1:length(model)-1]
    t = [Rational{BigInt}.(zeros(length(bias))) for bias in b]

    f_pre  = mlp_to_trop(w, b, t)[1]
    f_post = monomial_strong_elim(f_pre)

    return monomial_count(f_pre), monomial_count(f_post)
end

function run_experiment()
    widths = [2, 3, 4, 5, 6, 7, 8]
    num_trials = 3
    
    results = DataFrame(
        Width = Int[],
        Rate_Init = Float64[],
        Rate_Trained = Float64[]
    )

    X_train, X_test, Y_train, Y_test = generate_data()
    X_train_mat = Matrix(X_train')
    Y_train_mat = reshape(Y_train, 1, :)

    batch_size = 16
    lr = 1e-3
    max_epochs = 200

    loader = Flux.DataLoader(
        (X_train_mat, Y_train_mat),
        batchsize=batch_size,
        shuffle=true
    )

    loss(m, x, y) = Flux.binarycrossentropy(m(x), y)

    for width in widths
        println("Processing width: $width")
        
        sum_rate_init = 0.0
        sum_rate_trained = 0.0
        
        for trial in 1:num_trials
            # 1. Initialize
            w_init, b_init, t_init = random_mlp([2, width, 1])
            model = Chain(
                Dense(w_init[1], b_init[1], relu),
                Dense(w_init[2], b_init[2], identity),
                σ
            )

            # 2. Compute Init Rate
            pre_init, post_init = get_monomial_counts(model)
            sum_rate_init += (pre_init - post_init) / pre_init

            # 3. Train
            opt_state = Flux.setup(Flux.Adam(lr), model)
            for epoch in 1:max_epochs
                for (x_batch, y_batch) in loader
                    l, gs = Flux.withgradient(m -> loss(m, x_batch, y_batch), model)
                    Flux.update!(opt_state, model, gs[1])
                end
            end

            # 4. Compute Trained Rate
            pre_trained, post_trained = get_monomial_counts(model)
            sum_rate_trained += (pre_trained - post_trained) / pre_trained
        end
        
        # 5. Average the rates
        avg_rate_init = sum_rate_init / num_trials
        avg_rate_trained = sum_rate_trained / num_trials

        push!(results, (width, avg_rate_init, avg_rate_trained))
    end

    mkpath("outputs/rate_of_pruning")
    CSV.write("outputs/rate_of_pruning/results.csv", results)
end

run_experiment()