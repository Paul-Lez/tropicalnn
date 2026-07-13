include(joinpath(@__DIR__, "..", "experiment_setup.jl"))
const EXPERIMENT_RUNTIME = setup_experiment!()

using TropicalNN
using DataFrames
using CSV
using Logging
using Statistics

const REGION_MODE = highs_mode(EXPERIMENT_RUNTIME)
const WORKER_IDS = tropical_workers(EXPERIMENT_RUNTIME)

global_logger(SimpleLogger(stderr, Logging.Error))

# Random rational signomial: numerator and denominator have
# n_mons monomials with coefficients and exponents drawn uniformly from [0, 1].
function random_rational_signomial(n_vars, n_mons)
    num = Signomial(rand(Float64, n_mons), [rand(Float64, n_vars) for _ in 1:n_mons])
    den = Signomial(rand(Float64, n_mons), [rand(Float64, n_vars) for _ in 1:n_mons])
    return RationalSignomial(num, den)
end

function run_experiment()
    monomial_counts = [20, 50, 100, 200, 350, 500, 800, 1000]
    samples_per_nvars = [3 => 15, 4 => 15]

    output_dir = "outputs/random_rational_regions"
    mkpath(output_dir)
    results = DataFrame(NVars=Int[], NMonomials=Int[], Sample=Int[],
        NumRegions=Int[], Time=Float64[])

    for (n_vars, n_samples) in samples_per_nvars
        println("--- Linear regions of random rational signomials, $n_vars variables ---")
        for n_mons in monomial_counts
            println("Processing $n_mons monomials (num and den)")
            for s in 1:n_samples
                q = random_rational_signomial(n_vars, n_mons)
                try
                    t = @elapsed regions = TropicalNN.linear_regions(q;
                        mode=REGION_MODE, workers=WORKER_IDS)
                    push!(results, (n_vars, n_mons, s, length(regions), t))
                    println("  sample $s/$n_samples: $(length(regions)) regions",
                        " in $(round(t, digits=2)) s")
                catch err
                    println("  sample $s/$n_samples FAILED: $(sprint(showerror, err))")
                end
                # Checkpoint after every sample, like the original experiment
                CSV.write(joinpath(output_dir, "results.csv"), results)
            end
            write_averages(results, output_dir)
        end
    end
end

function write_averages(results, output_dir)
    grouped = groupby(results, [:NVars, :NMonomials])
    averages = combine(grouped,
        :NumRegions => mean => :MeanNumRegions,
        :Time => mean => :MeanTime,
        nrow => :NSamples)
    CSV.write(joinpath(output_dir, "averages.csv"), averages)
end

run_experiment()
