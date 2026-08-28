using CSV
using DataFrames
import Graphs
using JLD2
import Oscar
using Statistics
using TropicalNN

function domain_region_volumes(graph)
    region_volumes = Float64[]
    for vertex in Graphs.vertices(graph)
        region_volume = sum(Float64.(graph[vertex]["volume"]))
        isfinite(region_volume) && push!(region_volumes, region_volume)
    end
    return region_volumes
end

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
    mean_volumes = Float64[]
    median_volumes = Float64[]
    region_counts = Int[]
    total_volumes = Float64[]
    for step in steps
        checkpoint_path = joinpath(run_path, "checkpoints", lpad(string(step), 8, '0'))
        graph = JLD2.load(joinpath(checkpoint_path, "graph.jld2"))["graph"]
        volumes = domain_region_volumes(graph)
        isempty(volumes) && throw(ArgumentError(
            "the clipped analysis domain has no full-dimensional regions at step $step"
        ))
        push!(mean_volumes, mean(volumes))
        push!(median_volumes, median(volumes))
        push!(region_counts, length(volumes))
        push!(total_volumes, sum(volumes))
    end
    domain_volumes = DataFrame(
        Seed = fill(seed, length(steps)),
        Step = steps,
        Mean_Region_Volume = mean_volumes,
        Median_Region_Volume = median_volumes,
        Region_Count = region_counts,
        Total_Domain_Volume = total_volumes,
    )
    CSV.write(joinpath(run_path, "domain_volume_stats.csv"), domain_volumes)
    return metrics, final, monomials, domain_volumes
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

function summarize_domain_volumes(volumes)
    return combine(
        groupby(volumes, :Step),
        :Mean_Region_Volume => mean => :Mean_Region_Volume_Mean,
        :Mean_Region_Volume => _sample_std => :Mean_Region_Volume_Std,
        :Median_Region_Volume => mean => :Median_Region_Volume_Mean,
        :Median_Region_Volume => _sample_std => :Median_Region_Volume_Std,
        :Region_Count => mean => :Region_Count_Mean,
        :Region_Count => _sample_std => :Region_Count_Std,
        :Total_Domain_Volume => mean => :Total_Domain_Volume_Mean,
    )
end

function export_to_csvs(output_root)
    metric_frames = DataFrame[]
    final_frames = DataFrame[]
    monomial_frames = DataFrame[]
    volume_frames = DataFrame[]
    for run_path in run_directories(output_root)
        println("Exporting $(basename(run_path))")
        metrics, final, monomials, volumes = export_run_to_csvs(run_path)
        push!(metric_frames, metrics)
        push!(final_frames, final)
        push!(monomial_frames, monomials)
        push!(volume_frames, volumes)
    end

    if length(metric_frames) > 1
        all_metrics = vcat(metric_frames...)
        all_finals = vcat(final_frames...)
        all_monomials = vcat(monomial_frames...)
        all_volumes = vcat(volume_frames...)
        CSV.write(joinpath(output_root, "all_training_metrics.csv"), all_metrics)
        CSV.write(joinpath(output_root, "all_final_metrics.csv"), all_finals)
        CSV.write(joinpath(output_root, "all_monomial_counts.csv"), all_monomials)
        CSV.write(joinpath(output_root, "all_domain_volume_stats.csv"), all_volumes)
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
            joinpath(output_root, "domain_volume_summary.csv"),
            summarize_domain_volumes(all_volumes),
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
