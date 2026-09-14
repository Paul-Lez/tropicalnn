module PaperExports

using CSV
using DataFrames
using Dates
using Statistics

export export_paper_data,
       hoffman_paper_tables,
       mnist_paper_tables,
       width_depth_paper_table

_sample_std(values) = length(values) == 1 ? 0.0 : std(values)

function _require_columns(table, required, label)
    missing_columns = setdiff(required, propertynames(table))
    isempty(missing_columns) || throw(ArgumentError(
        "$label is missing columns: $(join(missing_columns, ", "))",
    ))
    return table
end

function _complete_groups(summary, label)
    isempty(summary) && throw(ArgumentError("$label has no rows"))
    sample_counts = unique(summary.NumSamples)
    length(sample_counts) == 1 || throw(ArgumentError(
        "$label has incomplete groups with sample counts $(sort(sample_counts))",
    ))
    return summary
end

function _require_unique_rows(table, columns, label)
    keys = Tuple.(eachrow(select(table, columns)))
    length(keys) == length(unique(keys)) || throw(ArgumentError(
        "$label contains duplicate rows for $(join(columns, ", "))",
    ))
    return table
end

function hoffman_paper_tables(samples::DataFrame)
    required = [
        :Benchmark,
        :MP,
        :MQ,
        :N,
        :Sample,
        :LowerHoffman,
        :LowerSeconds,
        :BruteForceHoffman,
        :BruteForceSeconds,
        :PVZHoffman,
        :PVZSeconds,
        :UpperHoffman,
        :UpperSeconds,
    ]
    _require_columns(samples, required, "Hoffman samples")
    _require_unique_rows(samples, [:MP, :MQ, :N, :Sample], "Hoffman samples")
    all(==("function_H_pq"), samples.Benchmark) || throw(ArgumentError(
        "paper Hoffman exports require function_H_pq samples",
    ))

    working = copy(samples)
    working.LowerAbsoluteError = abs.(
        working.LowerHoffman .- working.BruteForceHoffman,
    )
    working.BruteForceAbsoluteError = zeros(nrow(working))
    working.PVZAbsoluteError = abs.(
        working.PVZHoffman .- working.BruteForceHoffman,
    )
    working.UpperAbsoluteError = abs.(
        working.UpperHoffman .- working.BruteForceHoffman,
    )

    long = combine(
        groupby(working, [:MP, :MQ, :N]),
        :Sample => length => :NumSamples,
        :BruteForceHoffman => mean => :MeanBruteForceHoffman,
        :BruteForceHoffman => _sample_std => :StdBruteForceHoffman,
        :LowerAbsoluteError => mean => :MeanLowerAbsoluteError,
        :LowerAbsoluteError => _sample_std => :StdLowerAbsoluteError,
        :LowerSeconds => mean => :MeanLowerSeconds,
        :LowerSeconds => _sample_std => :StdLowerSeconds,
        :BruteForceSeconds => mean => :MeanBruteForceSeconds,
        :BruteForceSeconds => _sample_std => :StdBruteForceSeconds,
        :PVZAbsoluteError => mean => :MeanPVZAbsoluteError,
        :PVZAbsoluteError => _sample_std => :StdPVZAbsoluteError,
        :PVZSeconds => mean => :MeanPVZSeconds,
        :PVZSeconds => _sample_std => :StdPVZSeconds,
        :UpperAbsoluteError => mean => :MeanUpperAbsoluteError,
        :UpperAbsoluteError => _sample_std => :StdUpperAbsoluteError,
        :UpperSeconds => mean => :MeanUpperSeconds,
        :UpperSeconds => _sample_std => :StdUpperSeconds,
    )
    _complete_groups(long, "Hoffman samples")
    sort!(long, [:MP, :MQ, :N])

    quantities = [
        "Lower",
        "LowerTime",
        "BruteForce",
        "BruteForceTime",
        "PVZ",
        "PVZTime",
        "Upper",
        "UpperTime",
    ]
    wide = DataFrame(Quantity = quantities)
    for row in eachrow(long)
        column = Symbol("mp$(row.MP)_mq$(row.MQ)_n$(row.N)")
        wide[!, column] = [
            row.MeanLowerAbsoluteError,
            row.MeanLowerSeconds,
            0.0,
            row.MeanBruteForceSeconds,
            row.MeanPVZAbsoluteError,
            row.MeanPVZSeconds,
            row.MeanUpperAbsoluteError,
            row.MeanUpperSeconds,
        ]
    end
    return working, long, wide
end

function _mnist_region_summary(samples)
    return combine(
        groupby(samples, [
            :Model,
            :Activation,
            :Architecture,
            :HiddenLayers,
            :Width,
            :Pieces,
            :Algorithm,
            :Encoding,
        ]),
        :Sample => length => :NumSamples,
        :NumRegions => mean => :NumRegions,
        :NumRegions => _sample_std => :StdNumRegions,
        :TimeSeconds => mean => :TimeSeconds,
        :TimeSeconds => _sample_std => :StdTimeSeconds,
        :TimeSeconds => median => :MedianTimeSeconds,
        :TimeSeconds => minimum => :MinTimeSeconds,
        :TimeSeconds => maximum => :MaxTimeSeconds,
    )
end

function _mnist_metrics_summary(samples)
    return combine(
        groupby(samples, [
            :Model,
            :Activation,
            :Architecture,
            :HiddenLayers,
            :Width,
            :Pieces,
            :Epochs,
            :BatchSize,
            :LearningRate,
        ]),
        :Sample => length => :NumSamples,
        :TrainAccuracy => mean => :TrainAccuracy,
        :TrainAccuracy => _sample_std => :StdTrainAccuracy,
        :TestAccuracy => mean => :TestAccuracy,
        :TestAccuracy => _sample_std => :StdTestAccuracy,
    )
end

function _validate_mnist_architectures(regions, metrics)
    expected = Set((layers, width) for layers in 1:2 for width in 4:8)
    region_keys = Set(zip(regions.HiddenLayers, regions.Width))
    metric_keys = Set(zip(metrics.HiddenLayers, metrics.Width))
    region_keys == expected || throw(ArgumentError(
        "MNIST region samples do not contain exactly the plotted ReLU architectures",
    ))
    metric_keys == expected || throw(ArgumentError(
        "MNIST metric samples do not contain exactly the plotted ReLU architectures",
    ))
end

function mnist_paper_tables(
        region_samples::DataFrame,
        metric_samples::DataFrame;
        validate_architectures = true,
)
    _require_columns(
        region_samples,
        [:Model, :Sample, :Activation, :Architecture, :HiddenLayers, :Width,
            :Pieces, :NumRegions, :TimeSeconds, :Algorithm, :Encoding],
        "MNIST region samples",
    )
    _require_columns(
        metric_samples,
        [:Model, :Sample, :Activation, :Architecture, :HiddenLayers, :Width,
            :Pieces, :Epochs, :BatchSize, :LearningRate, :TrainAccuracy,
            :TestAccuracy],
        "MNIST metric samples",
    )
    relu_regions = filter(:Activation => ==("relu"), region_samples)
    relu_metrics = filter(:Activation => ==("relu"), metric_samples)
    _require_unique_rows(relu_regions, [:Model, :Sample], "MNIST region samples")
    _require_unique_rows(relu_metrics, [:Model, :Sample], "MNIST metric samples")
    region_sample_keys = Set(zip(relu_regions.Model, relu_regions.Sample))
    metric_sample_keys = Set(zip(relu_metrics.Model, relu_metrics.Sample))
    region_sample_keys == metric_sample_keys || throw(ArgumentError(
        "MNIST region and metric samples do not identify the same trained models",
    ))
    validate_architectures && _validate_mnist_architectures(relu_regions, relu_metrics)

    region_summary = _complete_groups(
        _mnist_region_summary(relu_regions),
        "MNIST region samples",
    )
    metric_summary = _complete_groups(
        _mnist_metrics_summary(relu_metrics),
        "MNIST metric samples",
    )
    nrow(region_summary) == length(unique(region_summary.Model)) ||
        throw(ArgumentError("MNIST region export has duplicate model rows"))
    nrow(metric_summary) == length(unique(metric_summary.Model)) ||
        throw(ArgumentError("MNIST metric export has duplicate model rows"))
    sort!(region_summary, [:HiddenLayers, :Width])
    sort!(metric_summary, [:HiddenLayers, :Width])
    return relu_regions, relu_metrics, region_summary, metric_summary
end

function _validate_width_depth_architectures(summary)
    expected = Set{Tuple{String, String, Int, Int}}()
    for network in ("relu", "maxout")
        for width in 10:10:60
            push!(expected, (network, "width", 1, width))
        end
        for depth in 1:6
            push!(expected, (network, "depth", depth, 10))
        end
    end
    observed = Set(zip(
        summary.Network,
        summary.Sweep,
        summary.HiddenLayers,
        summary.Width,
    ))
    observed == expected || throw(ArgumentError(
        "width/depth samples do not contain exactly the plotted architectures",
    ))
end

function width_depth_paper_table(samples::DataFrame; validate_architectures = true)
    _require_columns(
        samples,
        [:Network, :Sweep, :Architecture, :HiddenLayers, :Width, :Pieces,
            :Trial, :Seed, :NumRegions, :TimeSeconds, :Algorithm, :Encoding],
        "width/depth samples",
    )
    _require_unique_rows(
        samples,
        [:Network, :Sweep, :Architecture, :Trial],
        "width/depth samples",
    )
    summary = combine(
        groupby(samples, [
            :Network,
            :Sweep,
            :Architecture,
            :HiddenLayers,
            :Width,
            :Pieces,
            :Algorithm,
            :Encoding,
        ]),
        :Trial => length => :NumSamples,
        :NumRegions => mean => :NumRegions,
        :NumRegions => _sample_std => :StdNumRegions,
        :TimeSeconds => mean => :TimeSeconds,
        :TimeSeconds => _sample_std => :StdTimeSeconds,
        :TimeSeconds => median => :MedianTimeSeconds,
        :TimeSeconds => minimum => :MinTimeSeconds,
        :TimeSeconds => maximum => :MaxTimeSeconds,
    )
    _complete_groups(summary, "width/depth samples")
    validate_architectures && _validate_width_depth_architectures(summary)
    keys = zip(summary.Network, summary.Sweep, summary.Architecture)
    nrow(summary) == length(unique(keys)) ||
        throw(ArgumentError("width/depth export has duplicate architecture rows"))
    sort!(summary, [:Sweep, :Network, :HiddenLayers, :Width])
    return summary
end

function _fresh_output_directory(path)
    if ispath(path)
        isempty(readdir(path)) || throw(ArgumentError(
            "$path is not empty; choose a new export directory",
        ))
    else
        mkpath(path)
    end
    return path
end

function _write_table(path, table)
    mkpath(dirname(path))
    CSV.write(path, table)
    return path
end

function export_paper_data(;
        hoffman_samples_path,
        mnist_regions_path,
        mnist_metrics_path,
        width_depth_path,
        output_dir,
)
    _fresh_output_directory(output_dir)
    hoffman_samples = CSV.read(hoffman_samples_path, DataFrame)
    mnist_regions = CSV.read(mnist_regions_path, DataFrame)
    mnist_metrics = CSV.read(mnist_metrics_path, DataFrame)
    width_depth = CSV.read(width_depth_path, DataFrame)

    hoffman_raw, hoffman_long, hoffman_wide = hoffman_paper_tables(hoffman_samples)
    relu_regions, relu_metrics, mnist_region_summary, mnist_metric_summary =
        mnist_paper_tables(mnist_regions, mnist_metrics)
    width_depth_summary = width_depth_paper_table(width_depth)

    destinations = [
        _write_table(joinpath(output_dir, "hoffman-signomials", "hoffman_samples.csv"), hoffman_raw),
        _write_table(joinpath(output_dir, "hoffman-signomials", "hoffman_summary_long.csv"), hoffman_long),
        _write_table(joinpath(output_dir, "hoffman-signomials", "hoffman_summary.csv"), hoffman_wide),
        _write_table(joinpath(output_dir, "mnist", "linear_region_samples_all.csv"), mnist_regions),
        _write_table(joinpath(output_dir, "mnist", "metric_samples_all.csv"), mnist_metrics),
        _write_table(joinpath(output_dir, "mnist", "linear_region_samples.csv"), relu_regions),
        _write_table(joinpath(output_dir, "mnist", "metric_samples.csv"), relu_metrics),
        _write_table(joinpath(output_dir, "mnist", "linear_regions.csv"), mnist_region_summary),
        _write_table(joinpath(output_dir, "mnist", "metrics.csv"), mnist_metric_summary),
        _write_table(joinpath(output_dir, "width-depth-linear-regions", "linear_region_samples.csv"), width_depth),
        _write_table(joinpath(output_dir, "width-depth-linear-regions", "linear_regions.csv"), width_depth_summary),
    ]
    sources = [
        hoffman_samples_path,
        mnist_regions_path,
        mnist_metrics_path,
        width_depth_path,
    ]
    manifest = DataFrame(
        ExportedAtUTC = fill(string(Dates.now(Dates.UTC)), length(sources)),
        Source = abspath.(sources),
        SourceBytes = filesize.(sources),
    )
    _write_table(joinpath(output_dir, "export_manifest.csv"), manifest)
    return destinations
end

function _option(args, name, default)
    prefix = "$name="
    for (index, argument) in pairs(args)
        startswith(argument, prefix) && return argument[(lastindex(prefix) + 1):end]
        if argument == name
            index < length(args) || throw(ArgumentError("missing value after $name"))
            return args[index + 1]
        end
    end
    return default
end

function main(args = ARGS)
    input_root = _option(args, "--input-root", joinpath(@__DIR__, "outputs"))
    output_dir = _option(args, "--output", nothing)
    output_dir === nothing && throw(ArgumentError("--output is required"))
    hoffman_path = _option(
        args,
        "--hoffman-samples",
        joinpath(
            input_root,
            "effective_radius",
            "function_benchmark",
            "hoffman_samples.csv",
        ),
    )
    export_paper_data(
        hoffman_samples_path = hoffman_path,
        mnist_regions_path = joinpath(input_root, "mnist", "linear_regions.csv"),
        mnist_metrics_path = joinpath(input_root, "mnist", "metrics.csv"),
        width_depth_path = joinpath(input_root, "width_depth", "linear_regions.csv"),
        output_dir = output_dir,
    )
    println("Paper-ready CSVs written to $(abspath(output_dir))")
end

end


if abspath(PROGRAM_FILE) == @__FILE__
    PaperExports.main()
end
