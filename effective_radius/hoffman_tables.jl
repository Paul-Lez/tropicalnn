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
    return vec(TropicalNN._tilde_matrices((numerator_exponents, denominator_exponents)))
end

function sampled_lower_hoff(matrix, num_samples)
    return lower_hoffman_constant(matrix, num_samples)
end

function warm_up_algorithms()
    matrix = [1.0 0.0; 0.0 1.0; -1.0 -1.0]
    hoffman_constant(matrix; brute_force=true)
    hoffman_constant(matrix)
    sampled_lower_hoff(matrix, 1)
    upper_hoffman_constant(matrix)
    return nothing
end

function compute_table(config;
        num_samples = 30,
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
            matrices, matrix -> sampled_lower_hoff(matrix, lower_samples))
        brute, brute_time = matrix_statistics(
            matrices, matrix -> hoffman_constant(matrix; brute_force=true))
        pvz, pvz_time = matrix_statistics(matrices, hoffman_constant)
        upper, upper_time = matrix_statistics(matrices, upper_hoffman_constant)

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
    num_samples = parse(Int, option_value(args, "--hoffman-samples", "30"))
    lower_samples = parse(Int, option_value(args, "--hoffman-lower-samples", "100"))
    seed = parse(Int, option_value(args, "--hoffman-seed", "2024"))
    default_output = joinpath(@__DIR__, "..", "outputs", "effective_radius")
    output_dir = option_value(args, "--hoffman-output", default_output)

    warm_up_algorithms()
    Random.seed!(seed)
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
