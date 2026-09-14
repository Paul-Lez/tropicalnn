using Distributed

include(joinpath(@__DIR__, "..", "experiment_setup.jl"))
const EXPERIMENT_RUNTIME = setup_experiment!()
const HOFFMAN_WORKERS = tropical_workers(EXPERIMENT_RUNTIME)

using CSV
using DataFrames

include(joinpath(@__DIR__, "hoffman_summary.jl"))

@everywhere begin
    using LinearAlgebra
    using Random
    using Statistics
    using TropicalNN

    const LOWER_SAMPLE_POLICY =
        "ceil_one_tenth_brute_force_candidates_per_cell_matrix"

    function timed_value(f)
        value = nothing
        seconds = @elapsed value = f()
        return value, seconds
    end

    function matrix_collection_statistics(matrices, algorithm)
        values = Float64[]
        times = Float64[]
        for matrix in matrices
            value, seconds = timed_value(() -> algorithm(matrix))
            push!(values, value)
            push!(times, seconds)
        end
        return maximum(values), mean(times)
    end

    function random_rational_signomial(rng, m_p, m_q, n)
        numerator_exponents = rand(rng, m_p, n)
        denominator_exponents = rand(rng, m_q, n)
        numerator = Signomial(
            zeros(m_p),
            [collect(row) for row in eachrow(numerator_exponents)];
            sorted = false,
        )
        denominator = Signomial(
            zeros(m_q),
            [collect(row) for row in eachrow(denominator_exponents)];
            sorted = false,
        )
        return RationalSignomial(numerator, denominator)
    end

    function brute_force_candidate_count(matrix)
        count = 0
        for subset_size in 1:min(size(matrix)...)
            for subset in TropicalNN.Combinatorics.combinations(
                    1:size(matrix, 1),
                    subset_size,
            )
                LinearAlgebra.rank(matrix[subset, :]) == subset_size && (count += 1)
            end
        end
        return count
    end

    one_tenth_sample_count(candidate_count) = cld(candidate_count, 10)

    function lower_statistics(matrices)
        values = Float64[]
        times = Float64[]
        candidate_counts = Int[]
        sample_counts = Int[]
        for matrix in matrices
            candidate_count = brute_force_candidate_count(matrix)
            sample_count = one_tenth_sample_count(candidate_count)
            value, seconds = timed_value(
                () -> lower_hoffman_constant(matrix, sample_count),
            )
            push!(values, Float64(value))
            push!(times, seconds)
            push!(candidate_counts, candidate_count)
            push!(sample_counts, sample_count)
        end
        total_candidates = sum(candidate_counts)
        total_samples = sum(sample_counts)
        return (
            value = maximum(values; init = 0.0),
            mean_seconds = isempty(times) ? 0.0 : mean(times),
            candidate_count = total_candidates,
            sample_count = total_samples,
            sampling_fraction = iszero(total_candidates) ? 0.0 :
                total_samples / total_candidates,
            min_samples_per_matrix = isempty(sample_counts) ? 0 :
                minimum(sample_counts),
            max_samples_per_matrix = isempty(sample_counts) ? 0 :
                maximum(sample_counts),
            matrix_count = length(matrices),
        )
    end

    function function_lower_statistics(f, mode)
        filtered = TropicalNN._hoffman_nonzero_terms(f)
        exponent_matrices, _ = TropicalNN._linearmap_matrices(filtered)
        matrices = TropicalNN._hoffman_cell_matrices(
            filtered,
            exponent_matrices;
            mode = mode,
        )
        return lower_statistics(matrices)
    end

    function warm_up_algorithms(highs_threads = 1)
        matrix = [1.0 0.0; 0.0 1.0; -1.0 -1.0]
        hoffman_constant(matrix; brute_force=true)
        hoffman_constant(matrix)
        lower_hoffman_constant(matrix, one_tenth_sample_count(
            brute_force_candidate_count(matrix),
        ))
        upper_hoffman_constant(matrix)
        f = random_rational_signomial(Random.MersenneTwister(0), 2, 1, 2)
        mode = HiGHSMode(threads = highs_threads)
        Random.seed!(0)
        hoffman_constant(f; brute_force = true, mode = mode)
        hoffman_constant(f; mode = mode)
        function_lower_statistics(f, mode)
        upper_hoffman_constant(f; mode = mode)
        return nothing
    end

    function function_cell_matrix_count(f, mode)
        filtered = TropicalNN._hoffman_nonzero_terms(f)
        exponent_matrices, _ = TropicalNN._linearmap_matrices(filtered)
        return length(TropicalNN._hoffman_cell_matrices(
            filtered,
            exponent_matrices;
            mode = mode,
        ))
    end

    function compute_function_hoffman_sample(job)
        println("  sample $(job.sample)/$(job.num_samples)")
        f = random_rational_signomial(
            Random.MersenneTwister(job.function_seed),
            job.m_p,
            job.m_q,
            job.n,
        )
        mode = HiGHSMode(threads = job.highs_threads)
        Random.seed!(job.lower_seed)

        lower_result, lower_time = timed_value(
            () -> function_lower_statistics(f, mode),
        )
        brute, brute_time = timed_value(
            () -> hoffman_constant(f; brute_force = true, mode = mode),
        )
        pvz, pvz_time = timed_value(() -> hoffman_constant(f; mode = mode))
        upper, upper_time = timed_value(() -> upper_hoffman_constant(f; mode = mode))

        return (
            Benchmark = "function_H_pq",
            Sample = job.sample,
            FunctionSeed = job.function_seed,
            LowerSeed = job.lower_seed,
            LowerSamplePolicy = LOWER_SAMPLE_POLICY,
            BruteForceCandidates = lower_result.candidate_count,
            LowerSamples = lower_result.sample_count,
            LowerSamplingFraction = lower_result.sampling_fraction,
            MinLowerSamplesPerCellMatrix = lower_result.min_samples_per_matrix,
            MaxLowerSamplesPerCellMatrix = lower_result.max_samples_per_matrix,
            CoefficientPolicy = "all_zero",
            CellMatrices = lower_result.matrix_count,
            LowerHoffman = lower_result.value,
            LowerSeconds = lower_time,
            BruteForceHoffman = brute,
            BruteForceSeconds = brute_time,
            PVZHoffman = pvz,
            PVZSeconds = pvz_time,
            PVZSpeedup = brute_time / pvz_time,
            UpperHoffman = upper,
            UpperSeconds = upper_time,
        )
    end

    function compute_matrix_collection_sample(job)
        println("  all-pairs matrix sample $(job.sample)/$(job.num_samples)")
        rng = Random.MersenneTwister(job.function_seed)
        numerator_exponents = rand(rng, job.m_p, job.n)
        denominator_exponents = rand(rng, job.m_q, job.n)
        matrices = vec(TropicalNN._tilde_matrices((
            numerator_exponents,
            denominator_exponents,
        )))
        Random.seed!(job.lower_seed)
        lower_result = lower_statistics(matrices)
        brute, brute_time = matrix_collection_statistics(
            matrices,
            matrix -> hoffman_constant(matrix; brute_force = true),
        )
        pvz, pvz_time = matrix_collection_statistics(matrices, hoffman_constant)
        upper, upper_time = matrix_collection_statistics(
            matrices,
            upper_hoffman_constant,
        )
        return (
            Benchmark = "all_pairs_matrix_collection",
            Sample = job.sample,
            FunctionSeed = job.function_seed,
            LowerSeed = job.lower_seed,
            LowerSamplePolicy = LOWER_SAMPLE_POLICY,
            BruteForceCandidates = lower_result.candidate_count,
            LowerSamples = lower_result.sample_count,
            LowerSamplingFraction = lower_result.sampling_fraction,
            MinLowerSamplesPerCellMatrix = lower_result.min_samples_per_matrix,
            MaxLowerSamplesPerCellMatrix = lower_result.max_samples_per_matrix,
            CoefficientPolicy = "not_applicable",
            CellMatrices = length(matrices),
            LowerHoffman = lower_result.value,
            LowerSeconds = lower_result.mean_seconds,
            BruteForceHoffman = brute,
            BruteForceSeconds = brute_time,
            PVZHoffman = pvz,
            PVZSeconds = pvz_time,
            PVZSpeedup = brute_time / pvz_time,
            UpperHoffman = upper,
            UpperSeconds = upper_time,
        )
    end
end

const DEFAULT_CONFIGURATIONS = [
    (m_p = 2, m_q = 3, n = 6),
    (m_p = 3, m_q = 4, n = 9),
    (m_p = 5, m_q = 4, n = 8),
    (m_p = 7, m_q = 3, n = 12),
    # Fifteen constraints in two variables stress the PVZ frontier while
    # keeping exhaustive Hoffman enumeration small.
    (m_p = 7, m_q = 8, n = 2),
    (m_p = 6, m_q = 9, n = 2),
]

function empty_hoffman_results()
    return DataFrame(
        Benchmark = String[],
        MP = Int[],
        MQ = Int[],
        N = Int[],
        Sample = Int[],
        FunctionSeed = UInt64[],
        LowerSeed = UInt64[],
        LowerSamplePolicy = String[],
        BruteForceCandidates = Int[],
        LowerSamples = Int[],
        LowerSamplingFraction = Float64[],
        MinLowerSamplesPerCellMatrix = Int[],
        MaxLowerSamplesPerCellMatrix = Int[],
        CoefficientPolicy = String[],
        CellMatrices = Int[],
        LowerHoffman = Float64[],
        LowerSeconds = Float64[],
        BruteForceHoffman = Float64[],
        BruteForceSeconds = Float64[],
        PVZHoffman = Float64[],
        PVZSeconds = Float64[],
        PVZSpeedup = Float64[],
        UpperHoffman = Float64[],
        UpperSeconds = Float64[],
    )
end

function write_hoffman_results(output_dir, results, filename)
    output_path = joinpath(output_dir, filename)
    temporary_path = output_path * ".tmp"
    CSV.write(temporary_path, results)
    mv(temporary_path, output_path; force = true)
    return output_path
end

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
        rng = Random.default_rng(),
        lower_rng = Random.default_rng(),
        workers = nothing,
        benchmark = :function,
        highs_threads = 1,
)
    jobs = [(
        sample = sample,
        num_samples = num_samples,
        m_p = config.m_p,
        m_q = config.m_q,
        n = config.n,
        function_seed = rand(rng, UInt64),
        lower_seed = rand(lower_rng, UInt64),
        highs_threads = highs_threads,
    ) for sample in 1:num_samples]

    sample_function = if benchmark == :function
        compute_function_hoffman_sample
    elseif benchmark == :matrix_collection
        compute_matrix_collection_sample
    else
        throw(ArgumentError("unknown Hoffman benchmark: $benchmark"))
    end

    sample_results = if workers === nothing
        map(sample_function, jobs)
    else
        Distributed.pmap(sample_function, workers, jobs)
    end
    return DataFrame(sample_results)
end

function warm_up_sample_workers(workers, highs_threads)
    warm_up_algorithms(highs_threads)
    workers === nothing && return nothing

    @sync for pid in Distributed.workers(workers)
        @async Distributed.remotecall_wait(warm_up_algorithms, pid, highs_threads)
    end
    return nothing
end

function run_hoffman_tables(args = ARGS)
    num_samples = parse(Int, option_value(args, "--hoffman-samples", "30"))
    any(arg -> startswith(arg, "--hoffman-lower-samples"), args) && error(
        "--hoffman-lower-samples was removed; the lower budget is now one tenth " *
        "of each matrix's brute-force candidate count, rounded up",
    )
    seed = parse(Int, option_value(args, "--hoffman-seed", "2024"))
    default_output = joinpath(
        @__DIR__,
        "..",
        "outputs",
        "effective_radius",
        "function_benchmark",
    )
    output_dir = option_value(args, "--hoffman-output", default_output)
    configurations = selected_configurations(args)
    benchmark_name = option_value(args, "--hoffman-benchmark", "function")
    benchmark = if benchmark_name == "function"
        :function
    elseif benchmark_name == "matrix_collection"
        :matrix_collection
    else
        error("--hoffman-benchmark must be function or matrix_collection")
    end
    samples_filename = benchmark == :function ?
        "hoffman_samples.csv" : "all_pairs_matrix_samples.csv"
    summary_filename = benchmark == :function ?
        "hoffman_summary.csv" : "all_pairs_matrix_summary.csv"
    existing_outputs = filter(isfile, [
        joinpath(output_dir, samples_filename),
        joinpath(output_dir, summary_filename),
    ])
    isempty(existing_outputs) || error(
        "refusing to overwrite Hoffman outputs; choose a new --hoffman-output directory: " *
        join(existing_outputs, ", "),
    )

    warm_up_sample_workers(HOFFMAN_WORKERS, EXPERIMENT_RUNTIME.highs_threads)
    rng = MersenneTwister(seed)
    lower_rng = MersenneTwister(seed)
    mkpath(output_dir)

    if HOFFMAN_WORKERS !== nothing
        println("Running Hoffman samples on ",
            length(Distributed.workers(HOFFMAN_WORKERS)), " workers")
    end

    results = empty_hoffman_results()
    for config in configurations
        println("Hoffman table: m_p=$(config.m_p), m_q=$(config.m_q), n=$(config.n)")
        table = compute_table(config;
            num_samples = num_samples,
            rng = rng,
            lower_rng = lower_rng,
            workers = HOFFMAN_WORKERS,
            benchmark = benchmark,
            highs_threads = EXPERIMENT_RUNTIME.highs_threads,
        )
        insertcols!(
            table,
            1,
            :MP => fill(config.m_p, nrow(table)),
            :MQ => fill(config.m_q, nrow(table)),
            :N => fill(config.n, nrow(table)),
        )
        append!(results, table; cols = :setequal, promote = false)
        results_path = write_hoffman_results(output_dir, results, samples_filename)
        summary_path = write_hoffman_summary(
            output_dir,
            results,
            num_samples;
            filename = summary_filename,
        )
        println("Saved $results_path")
        println("Saved $summary_path")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_hoffman_tables()
end
