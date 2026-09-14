using CSV
using DataFrames
using Statistics

const HOFFMAN_MEASUREMENT_COLUMNS = [
    :LowerHoffman,
    :LowerMeanSeconds,
    :BruteForceHoffman,
    :BruteForceMeanSeconds,
    :PVZHoffman,
    :PVZMeanSeconds,
    :ExactAbsoluteDifference,
    :PVZSpeedup,
    :UpperHoffman,
    :UpperMeanSeconds,
]

function hoffman_summary(samples, num_samples)
    transformations = Pair[]
    for column in HOFFMAN_MEASUREMENT_COLUMNS
        push!(transformations, column => mean => Symbol("Mean", column))
        push!(transformations, column => std => Symbol("Std", column))
    end

    summary = combine(
        groupby(samples, [:MP, :MQ, :N]),
        :Sample => length => :NumSamples,
        transformations...,
    )
    filter!(:NumSamples => ==(num_samples), summary)
    return summary
end

function write_hoffman_summary(output_dir, samples, num_samples)
    output_path = joinpath(output_dir, "hoffman_summary.csv")
    temporary_path = output_path * ".tmp"
    CSV.write(temporary_path, hoffman_summary(samples, num_samples))
    mv(temporary_path, output_path; force = true)
    return output_path
end
