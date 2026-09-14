using CSV
using DataFrames
using Statistics

const HOFFMAN_MEASUREMENT_COLUMNS = [
    :LowerHoffman,
    :LowerSeconds,
    :BruteForceHoffman,
    :BruteForceSeconds,
    :PVZHoffman,
    :PVZSeconds,
    :LowerAbsoluteError,
    :PVZAbsoluteError,
    :PVZSpeedup,
    :UpperHoffman,
    :UpperSeconds,
    :UpperAbsoluteError,
]

function hoffman_summary(samples, num_samples)
    working = copy(samples)
    working.LowerAbsoluteError = abs.(
        working.LowerHoffman .- working.BruteForceHoffman,
    )
    working.PVZAbsoluteError = abs.(
        working.PVZHoffman .- working.BruteForceHoffman,
    )
    working.UpperAbsoluteError = abs.(
        working.UpperHoffman .- working.BruteForceHoffman,
    )
    transformations = Pair[]
    for column in HOFFMAN_MEASUREMENT_COLUMNS
        push!(transformations, column => mean => Symbol("Mean", column))
        push!(transformations, column => std => Symbol("Std", column))
    end

    summary = combine(
        groupby(working, [
            :Benchmark,
            :MP,
            :MQ,
            :N,
            :CoefficientPolicy,
            :LowerSamplePolicy,
        ]),
        :Sample => length => :NumSamples,
        :CellMatrices => mean => :MeanCellMatrices,
        :CellMatrices => std => :StdCellMatrices,
        :BruteForceCandidates => mean => :MeanBruteForceCandidates,
        :BruteForceCandidates => std => :StdBruteForceCandidates,
        :LowerSamples => mean => :MeanLowerSamples,
        :LowerSamples => std => :StdLowerSamples,
        :LowerSamplingFraction => mean => :MeanLowerSamplingFraction,
        :LowerSamplingFraction => std => :StdLowerSamplingFraction,
        :MinLowerSamplesPerCellMatrix => minimum =>
            :MinLowerSamplesPerCellMatrix,
        :MaxLowerSamplesPerCellMatrix => maximum =>
            :MaxLowerSamplesPerCellMatrix,
        transformations...,
    )
    filter!(:NumSamples => ==(num_samples), summary)
    return summary
end

function write_hoffman_summary(output_dir, samples, num_samples;
        filename = "hoffman_summary.csv")
    output_path = joinpath(output_dir, filename)
    temporary_path = output_path * ".tmp"
    CSV.write(temporary_path, hoffman_summary(samples, num_samples))
    mv(temporary_path, output_path; force = true)
    return output_path
end
