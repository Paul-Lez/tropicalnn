using CSV
using DataFrames
import Graphs
using JLD2
using Statistics
using TropicalNN

graph_region_volumes(graph) = [
    sum(Float64.(graph[vertex]["volume"])) for vertex in Graphs.vertices(graph)
]

function checkpoint_steps(run_path)
    checkpoint_root = joinpath(run_path, "checkpoints")
    isdir(checkpoint_root) || return Int[]
    names = filter(readdir(checkpoint_root)) do name
        occursin(r"^\d+$", name) && isdir(joinpath(checkpoint_root, name))
    end
    return sort(parse.(Int, names))
end

function run_directories(output_root)
    isfile(joinpath(output_root, "config.jld2")) && return [output_root]
    isdir(output_root) || throw(ArgumentError("output directory does not exist: $output_root"))
    paths = filter(readdir(output_root; join = true)) do path
        isdir(path) && occursin(r"^seed_\d+$", basename(path)) &&
            isfile(joinpath(path, "config.jld2"))
    end
    isempty(paths) && throw(ArgumentError("no seed runs found in $output_root"))
    return sort(paths; by = path -> parse(Int, split(basename(path), '_')[end]))
end

function export_run_to_csvs(run_path)
    config = JLD2.load(joinpath(run_path, "config.jld2"))["config_data"]
    get(config, "analysis_domain", nothing) == "whole_plane" || throw(ArgumentError(
        "run is not marked as a whole-plane analysis: $run_path"
    ))
    training = JLD2.load(joinpath(run_path, "training_data.jld2"))["training_data"]
    final_metrics = JLD2.load(joinpath(run_path, "final_metrics.jld2"))["final_metrics"]
    seed = Int(config["seed"])

    metrics = DataFrame(
        Seed = fill(seed, length(training["step"])),
        Step = training["step"],
        Examples_Seen = training["examples_seen"],
        Equivalent_Epoch = training["equivalent_epochs"],
        Train_Loss = training["train_loss"],
        Validation_Loss = training["validation_loss"],
        Train_Accuracy = training["train_accuracy"],
        Validation_Accuracy = training["validation_accuracy"],
        Parameter_Norm = training["parameter_norm"],
    )
    CSV.write(joinpath(run_path, "training_metrics.csv"), metrics)

    final = DataFrame(
        Seed = [seed],
        Step = [Int(final_metrics["step"])],
        Train_Loss = [last(training["train_loss"])],
        Validation_Loss = [last(training["validation_loss"])],
        Test_Loss = [Float64(final_metrics["test_loss"])],
        Train_Accuracy = [last(training["train_accuracy"])],
        Validation_Accuracy = [last(training["validation_accuracy"])],
        Test_Accuracy = [Float64(final_metrics["test_accuracy"])],
    )
    CSV.write(joinpath(run_path, "final_metrics.csv"), final)

    monomial_data = JLD2.load(joinpath(run_path, "monomial_data.jld2"))["monomial_data"]
    monomials = DataFrame(
        Seed = fill(seed, length(monomial_data["step"])),
        Step = monomial_data["step"],
        Pre_Pruning = monomial_data["pre"],
        Post_Pruning = monomial_data["post"],
    )
    CSV.write(joinpath(run_path, "monomial_counts.csv"), monomials)

    steps = checkpoint_steps(run_path)
    mean_finite_volumes = Float64[]
    median_finite_volumes = Float64[]
    finite_region_counts = Int[]
    unbounded_region_counts = Int[]
    total_region_counts = Int[]
    for step in steps
        checkpoint_path = joinpath(run_path, "checkpoints", lpad(string(step), 8, '0'))
        graph = JLD2.load(joinpath(checkpoint_path, "graph.jld2"))["graph"]
        region_volumes = graph_region_volumes(graph)
        isempty(region_volumes) && throw(ArgumentError(
            "the whole-plane subdivision has no full-dimensional regions at step $step"
        ))
        all(volume -> isfinite(volume) || isinf(volume), region_volumes) ||
            throw(ArgumentError("invalid whole-plane region volume at step $step"))
        finite_volumes = filter(isfinite, region_volumes)
        push!(mean_finite_volumes, isempty(finite_volumes) ? NaN : mean(finite_volumes))
        push!(median_finite_volumes, isempty(finite_volumes) ? NaN : median(finite_volumes))
        push!(finite_region_counts, length(finite_volumes))
        push!(unbounded_region_counts, count(isinf, region_volumes))
        push!(total_region_counts, length(region_volumes))
    end
    region_stats = DataFrame(
        Seed = fill(seed, length(steps)),
        Step = steps,
        Mean_Finite_Region_Volume = mean_finite_volumes,
        Median_Finite_Region_Volume = median_finite_volumes,
        Finite_Region_Count = finite_region_counts,
        Unbounded_Region_Count = unbounded_region_counts,
        Total_Region_Count = total_region_counts,
    )
    CSV.write(joinpath(run_path, "whole_plane_region_stats.csv"), region_stats)
    return metrics, final, monomials, region_stats
end

_sample_std(values) = length(values) == 1 ? 0.0 : std(values)

function summarize_training(metrics)
    return combine(
        groupby(metrics, :Step),
        :Equivalent_Epoch => mean => :Equivalent_Epoch,
        :Train_Loss => mean => :Train_Loss_Mean,
        :Train_Loss => _sample_std => :Train_Loss_Std,
        :Validation_Loss => mean => :Validation_Loss_Mean,
        :Validation_Loss => _sample_std => :Validation_Loss_Std,
        :Train_Accuracy => mean => :Train_Accuracy_Mean,
        :Train_Accuracy => _sample_std => :Train_Accuracy_Std,
        :Validation_Accuracy => mean => :Validation_Accuracy_Mean,
        :Validation_Accuracy => _sample_std => :Validation_Accuracy_Std,
        :Parameter_Norm => mean => :Parameter_Norm_Mean,
        :Parameter_Norm => _sample_std => :Parameter_Norm_Std,
    )
end

function summarize_final_metrics(final_metrics)
    steps = unique(final_metrics.Step)
    length(steps) == 1 || throw(ArgumentError("final metrics use different step counts"))
    return DataFrame(
        Runs = [nrow(final_metrics)],
        Step = [only(steps)],
        Train_Loss_Mean = [mean(final_metrics.Train_Loss)],
        Train_Loss_Std = [_sample_std(final_metrics.Train_Loss)],
        Validation_Loss_Mean = [mean(final_metrics.Validation_Loss)],
        Validation_Loss_Std = [_sample_std(final_metrics.Validation_Loss)],
        Test_Loss_Mean = [mean(final_metrics.Test_Loss)],
        Test_Loss_Std = [_sample_std(final_metrics.Test_Loss)],
        Train_Accuracy_Mean = [mean(final_metrics.Train_Accuracy)],
        Train_Accuracy_Std = [_sample_std(final_metrics.Train_Accuracy)],
        Validation_Accuracy_Mean = [mean(final_metrics.Validation_Accuracy)],
        Validation_Accuracy_Std = [_sample_std(final_metrics.Validation_Accuracy)],
        Test_Accuracy_Mean = [mean(final_metrics.Test_Accuracy)],
        Test_Accuracy_Std = [_sample_std(final_metrics.Test_Accuracy)],
    )
end

function summarize_monomials(monomials)
    return combine(
        groupby(monomials, :Step),
        :Pre_Pruning => mean => :Pre_Pruning_Mean,
        :Pre_Pruning => _sample_std => :Pre_Pruning_Std,
        :Post_Pruning => mean => :Post_Pruning_Mean,
        :Post_Pruning => _sample_std => :Post_Pruning_Std,
    )
end


_finite_mean(values) = begin
    finite_values = filter(isfinite, values)
    isempty(finite_values) ? NaN : mean(finite_values)
end
_finite_std(values) = begin
    finite_values = filter(isfinite, values)
    isempty(finite_values) ? NaN : _sample_std(finite_values)
end

function summarize_region_volumes(volumes)
    return combine(
        groupby(volumes, :Step),
        :Mean_Finite_Region_Volume => _finite_mean => :Mean_Finite_Region_Volume_Mean,
        :Mean_Finite_Region_Volume => _finite_std => :Mean_Finite_Region_Volume_Std,
        :Median_Finite_Region_Volume => _finite_mean => :Median_Finite_Region_Volume_Mean,
        :Median_Finite_Region_Volume => _finite_std => :Median_Finite_Region_Volume_Std,
        :Finite_Region_Count => mean => :Finite_Region_Count_Mean,
        :Finite_Region_Count => _sample_std => :Finite_Region_Count_Std,
        :Unbounded_Region_Count => mean => :Unbounded_Region_Count_Mean,
        :Unbounded_Region_Count => _sample_std => :Unbounded_Region_Count_Std,
        :Total_Region_Count => mean => :Total_Region_Count_Mean,
        :Total_Region_Count => _sample_std => :Total_Region_Count_Std,
    )
end

function export_to_csvs(output_root)
    metric_frames = DataFrame[]
    final_frames = DataFrame[]
    monomial_frames = DataFrame[]
    region_frames = DataFrame[]
    for run_path in run_directories(output_root)
        println("Exporting $(basename(run_path))")
        metrics, final, monomials, regions = export_run_to_csvs(run_path)
        push!(metric_frames, metrics)
        push!(final_frames, final)
        push!(monomial_frames, monomials)
        push!(region_frames, regions)
    end

    if length(metric_frames) > 1
        all_metrics = vcat(metric_frames...)
        all_finals = vcat(final_frames...)
        all_monomials = vcat(monomial_frames...)
        all_regions = vcat(region_frames...)
        CSV.write(joinpath(output_root, "all_training_metrics.csv"), all_metrics)
        CSV.write(joinpath(output_root, "all_final_metrics.csv"), all_finals)
        CSV.write(joinpath(output_root, "all_monomial_counts.csv"), all_monomials)
        CSV.write(joinpath(output_root, "all_whole_plane_region_stats.csv"), all_regions)
        CSV.write(
            joinpath(output_root, "training_metrics_summary.csv"),
            summarize_training(all_metrics),
        )
        CSV.write(
            joinpath(output_root, "final_metrics_summary.csv"),
            summarize_final_metrics(all_finals),
        )
        CSV.write(
            joinpath(output_root, "monomial_counts_summary.csv"),
            summarize_monomials(all_monomials),
        )
        CSV.write(
            joinpath(output_root, "whole_plane_region_summary.csv"),
            summarize_region_volumes(all_regions),
        )
    end
    println("CSV export complete")
end

function main()
    default_dataset = lowercase(get(ENV, "VOLUME_DATASET", "spiral"))
    default_root = joinpath("outputs", "volume_dynamics", default_dataset)
    output_root = isempty(ARGS) ? get(ENV, "VOLUME_OUTPUT_DIR", default_root) : only(ARGS)
    export_to_csvs(output_root)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
