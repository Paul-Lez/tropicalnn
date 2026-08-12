using Distributed

include(joinpath(@__DIR__, "..", "experiment_setup.jl"))
const EXPERIMENT_RUNTIME = setup_experiment!()
const HOFFMAN_WORKERS = tropical_workers(EXPERIMENT_RUNTIME)

using CSV
using DataFrames

@everywhere begin
    using LinearAlgebra
    using Random
    using Statistics
    using TropicalNN

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

    function compute_hoffman_sample(job)
        println("  sample $(job.sample)/$(job.num_samples)")
        Random.seed!(job.lower_seed)

        lower, lower_time = matrix_statistics(
            job.matrices, matrix -> sampled_lower_hoff(matrix, job.lower_samples))
        brute, brute_time = matrix_statistics(
            job.matrices, matrix -> hoffman_constant(matrix; brute_force=true))
        pvz, pvz_time = matrix_statistics(job.matrices, hoffman_constant)
        upper, upper_time = matrix_statistics(job.matrices, upper_hoffman_constant)

        return (
            Sample = job.sample,
            LowerHoffman = lower,
            LowerMeanSeconds = lower_time,
            BruteForceHoffman = brute,
            BruteForceMeanSeconds = brute_time,
            PVZHoffman = pvz,
            PVZMeanSeconds = pvz_time,
            ExactAbsoluteDifference = abs(brute - pvz),
            PVZSpeedup = brute_time / pvz_time,
            UpperHoffman = upper,
            UpperMeanSeconds = upper_time,
        )
    end
end

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

function compute_table(config;
        num_samples = 30,
        lower_samples = 100,
        rng = Random.default_rng(),
        lower_rng = Random.default_rng(),
        workers = nothing,
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

    jobs = [(
        sample = sample,
        num_samples = num_samples,
        matrices = random_tilde_matrices(rng, config.m_p, config.m_q, config.n),
        lower_samples = lower_samples,
        lower_seed = rand(lower_rng, UInt64),
    ) for sample in 1:num_samples]

    sample_results = if workers === nothing
        map(compute_hoffman_sample, jobs)
    else
        Distributed.pmap(compute_hoffman_sample, workers, jobs)
    end

    for result in sample_results
        push!(results, result)
    end
    return results
end

function warm_up_sample_workers(workers)
    warm_up_algorithms()
    workers === nothing && return nothing

    @sync for pid in Distributed.workers(workers)
        @async Distributed.remotecall_wait(warm_up_algorithms, pid)
    end
    return nothing
end

function run_hoffman_tables(args = ARGS)
    num_samples = parse(Int, option_value(args, "--hoffman-samples", "30"))
    lower_samples = parse(Int, option_value(args, "--hoffman-lower-samples", "100"))
    seed = parse(Int, option_value(args, "--hoffman-seed", "2024"))
    default_output = joinpath(@__DIR__, "..", "outputs", "effective_radius")
    output_dir = option_value(args, "--hoffman-output", default_output)

    warm_up_sample_workers(HOFFMAN_WORKERS)
    rng = MersenneTwister(seed)
    lower_rng = MersenneTwister(seed)
    mkpath(output_dir)

    if HOFFMAN_WORKERS !== nothing
        println("Running Hoffman samples on ",
            length(Distributed.workers(HOFFMAN_WORKERS)), " workers")
    end

    for config in selected_configurations(args)
        println("Hoffman table: m_p=$(config.m_p), m_q=$(config.m_q), n=$(config.n)")
        results = compute_table(config;
            num_samples = num_samples,
            lower_samples = lower_samples,
            rng = rng,
            lower_rng = lower_rng,
            workers = HOFFMAN_WORKERS,
        )
        filename = "table_mp$(config.m_p)_mq$(config.m_q)_n$(config.n).csv"
        output_path = joinpath(output_dir, filename)
        CSV.write(output_path, results)
        println("Saved $output_path")
    end
end

run_hoffman_tables()
