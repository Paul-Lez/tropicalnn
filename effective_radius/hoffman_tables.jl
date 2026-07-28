using CSV
using DataFrames
using LinearAlgebra
using Random
using Statistics
using TropicalNN

const DEFAULT_CONFIGURATIONS = [
    (m_p = 2, m_q = 3, n = 6),
    (m_p = 3, m_q = 4, n = 9),
    (m_p = 5, m_q = 4, n = 8),
    (m_p = 7, m_q = 3, n = 12),
]

function option_value(args, name, default)
    prefix = "$name="
    for (index, arg) in pairs(args)
        startswith(arg, prefix) && return arg[(lastindex(prefix) + 1):end]
        arg == name && return args[index + 1]
    end
    return default
end

function selected_configurations(args)
    requested = option_value(args, "--hoffman-config", nothing)
    requested === nothing && return DEFAULT_CONFIGURATIONS

    values = parse.(Int, split(requested, ','))
    length(values) == 3 ||
        error("--hoffman-config must have the form m_p,m_q,n")
    return [(m_p = values[1], m_q = values[2], n = values[3])]
end

function timed_value(f)
    value = nothing
    seconds = @elapsed value = f()
    return Float64(value), seconds
end

function matrix_statistics(matrices, algorithm)
    values = Float64[]
    times = Float64[]
    for matrix in matrices
        value, seconds = timed_value(() -> algorithm(matrix))
        push!(values, value)
        push!(times, seconds)
    end
    return maximum(values), mean(times)
end

function random_tilde_matrices(rng, m_p, m_q, n)
    numerator_exponents = rand(rng, m_p, n)
    denominator_exponents = rand(rng, m_q, n)
    return vec(tilde_matrices((numerator_exponents, denominator_exponents)))
end

function sampled_lower_hoff(rng, matrix, num_samples)
    num_rows = size(matrix, 1)
    lower = 0.0
    for _ in 1:num_samples
        subset_size = rand(rng, 1:num_rows)
        subset = randperm(rng, num_rows)[1:subset_size]
        _, optimum = surjectivity_test(matrix[subset, :])
        optimum > 0 && (lower = max(lower, 1 / optimum))
    end
    return lower
end

function warm_up_algorithms()
    matrix = [1.0 0.0; 0.0 1.0; -1.0 -1.0]
    exact_hoff(matrix)
    pvz_hoff(matrix)
    sampled_lower_hoff(MersenneTwister(1), matrix, 1)
    upper_hoff(matrix)
    return nothing
end

function compute_table(config;
        num_samples = 15,
        lower_samples = 100,
        rng = Random.default_rng(),
)
    results = DataFrame(
        Sample = Int[],
        LowerHoffman = Float64[],
        LowerMeanSeconds = Float64[],
        BruteForceHoffman = Float64[],
        BruteForceMeanSeconds = Float64[],
        PVZHoffman = Float64[],
        PVZMeanSeconds = Float64[],
        ExactAbsoluteDifference = Float64[],
        PVZSpeedup = Float64[],
        UpperHoffman = Float64[],
        UpperMeanSeconds = Float64[],
    )

    for sample in 1:num_samples
        println("  sample $sample/$num_samples")
        matrices = random_tilde_matrices(rng, config.m_p, config.m_q, config.n)

        lower, lower_time = matrix_statistics(
            matrices, matrix -> sampled_lower_hoff(rng, matrix, lower_samples))
        brute, brute_time = matrix_statistics(matrices, exact_hoff)
        pvz, pvz_time = matrix_statistics(matrices, pvz_hoff)
        upper, upper_time = matrix_statistics(matrices, upper_hoff)

        push!(results, (
            sample,
            lower,
            lower_time,
            brute,
            brute_time,
            pvz,
            pvz_time,
            abs(brute - pvz),
            brute_time / pvz_time,
            upper,
            upper_time,
        ))
    end
    return results
end

function run_hoffman_tables(args = ARGS)
    num_samples = parse(Int, option_value(args, "--hoffman-samples", "15"))
    lower_samples = parse(Int, option_value(args, "--hoffman-lower-samples", "100"))
    seed = parse(Int, option_value(args, "--hoffman-seed", "2024"))
    default_output = joinpath(@__DIR__, "..", "outputs", "effective_radius")
    output_dir = option_value(args, "--hoffman-output", default_output)

    num_samples > 0 || error("--hoffman-samples must be positive")
    lower_samples > 0 || error("--hoffman-lower-samples must be positive")

    warm_up_algorithms()
    rng = MersenneTwister(seed)
    mkpath(output_dir)

    for config in selected_configurations(args)
        println("Hoffman table: m_p=$(config.m_p), m_q=$(config.m_q), n=$(config.n)")
        results = compute_table(config;
            num_samples = num_samples,
            lower_samples = lower_samples,
            rng = rng,
        )
        filename = "table_mp$(config.m_p)_mq$(config.m_q)_n$(config.n).csv"
        output_path = joinpath(output_dir, filename)
        CSV.write(output_path, results)
        println("Saved $output_path")
    end
end

run_hoffman_tables()
