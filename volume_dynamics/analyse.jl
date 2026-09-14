using CSV
using DataFrames
import Graphs
using JLD2
using Statistics
using TropicalNN

graph_region_volumes(graph) = [
    sum(Float64.(graph[vertex]["volume"])) for vertex in Graphs.vertices(graph)
]

function checkpoint_epochs(run_path)
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
    epochs = checkpoint_epochs(run_path)
    isempty(epochs) && throw(ArgumentError("no epoch checkpoints found in $run_path"))
    training["epoch"] == epochs || throw(ArgumentError(
        "training measurements do not match epoch checkpoint directories",
    ))
    Int(final_metrics["epoch"]) == last(epochs) || throw(ArgumentError(
        "final metrics do not match the last checkpoint epoch",
    ))
    Int(final_metrics["step"]) == last(training["step"]) || throw(ArgumentError(
        "final metrics do not match the last checkpoint step",
    ))

    metrics = DataFrame(
        Seed = fill(seed, length(training["epoch"])),
        Epoch = training["epoch"],
        Step = training["step"],
        Examples_Seen = training["examples_seen"],
        Train_Loss = training["train_loss"],
        Validation_Loss = training["validation_loss"],
        Train_Accuracy = training["train_accuracy"],
        Validation_Accuracy = training["validation_accuracy"],
        Parameter_Norm = training["parameter_norm"],
    )
    CSV.write(joinpath(run_path, "training_metrics.csv"), metrics)

    final = DataFrame(
        Seed = [seed],
        Epoch = [Int(final_metrics["epoch"])],
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
    monomial_data["epoch"] == epochs || throw(ArgumentError(
        "monomial measurements do not match epoch checkpoint directories",
    ))
    monomial_data["step"] == training["step"] || throw(ArgumentError(
        "monomial measurements do not match training optimizer steps",
    ))
    monomials = DataFrame(
        Seed = fill(seed, length(monomial_data["epoch"])),
        Epoch = monomial_data["epoch"],
        Step = monomial_data["step"],
        Pre_Pruning = monomial_data["pre"],
        Post_Pruning = monomial_data["post"],
    )
    CSV.write(joinpath(run_path, "monomial_counts.csv"), monomials)

    checkpoint_data = JLD2.load(
        joinpath(run_path, "checkpoint_data.jld2"),
    )["checkpoint_data"]
    checkpoint_data["epoch"] == epochs || throw(ArgumentError(
        "checkpoint Hoffman measurements do not match epoch checkpoint directories",
    ))
    checkpoint_data["step"] == training["step"] || throw(ArgumentError(
        "checkpoint Hoffman measurements do not match training optimizer steps",
    ))
    hoffman = DataFrame(
        Seed = fill(seed, length(checkpoint_data["epoch"])),
        Epoch = checkpoint_data["epoch"],
        Step = checkpoint_data["step"],
        Hoffman_Constant = checkpoint_data["hoffman_constant"],
        Norm = fill(checkpoint_data["hoffman_norm"], length(checkpoint_data["step"])),
        Algorithm = fill(
            checkpoint_data["hoffman_algorithm"],
            length(checkpoint_data["step"]),
        ),
        Representation = fill(
            checkpoint_data["hoffman_representation"],
            length(checkpoint_data["step"]),
        ),
        Numerical_Evaluation = fill(
            checkpoint_data["hoffman_evaluation"],
            length(checkpoint_data["step"]),
        ),
    )
    CSV.write(joinpath(run_path, "hoffman_constants.csv"), hoffman)

    steps = training["step"]
    mean_finite_volumes = Float64[]
    median_finite_volumes = Float64[]
    finite_region_counts = Int[]
    unbounded_region_counts = Int[]
    total_region_counts = Int[]
    polyhedron_counts = Int[]
    mean_vertices_per_region = Float64[]
    max_vertices_per_region = Int[]
    mean_vertices_per_polyhedron = Float64[]
    max_vertices_per_polyhedron = Int[]
    mean_facets_per_polyhedron = Float64[]
    max_facets_per_polyhedron = Int[]
    mean_rays_per_polyhedron = Float64[]
    for (epoch, step) in zip(epochs, steps)
        checkpoint_path = joinpath(run_path, "checkpoints", lpad(string(epoch), 8, '0'))
        graph = JLD2.load(joinpath(checkpoint_path, "graph.jld2"))["graph"]
        region_volumes = graph_region_volumes(graph)
        isempty(region_volumes) && throw(ArgumentError(
            "the whole-plane subdivision has no full-dimensional regions at epoch $epoch"
        ))
        all(volume -> isfinite(volume) || isinf(volume), region_volumes) ||
            throw(ArgumentError("invalid whole-plane region volume at epoch $epoch"))
        finite_volumes = filter(isfinite, region_volumes)
        push!(mean_finite_volumes, isempty(finite_volumes) ? NaN : mean(finite_volumes))
        push!(median_finite_volumes, isempty(finite_volumes) ? NaN : median(finite_volumes))
        push!(finite_region_counts, length(finite_volumes))
        push!(unbounded_region_counts, count(isinf, region_volumes))
        push!(total_region_counts, length(region_volumes))

        polyhedral_data = JLD2.load(
            joinpath(checkpoint_path, "polyhedral_data.jld2"),
        )["polyhedral_data"]
        vertices_per_region = polyhedral_data["vertices_per_region"]
        vertices_per_polyhedron = polyhedral_data["vertices_per_polyhedron"]
        facets_per_polyhedron = polyhedral_data["facets_per_polyhedron"]
        rays_per_polyhedron = polyhedral_data["rays_per_polyhedron"]
        push!(polyhedron_counts, sum(polyhedral_data["polyhedra_per_region"]))
        push!(mean_vertices_per_region, mean(vertices_per_region))
        push!(max_vertices_per_region, maximum(vertices_per_region))
        push!(mean_vertices_per_polyhedron, mean(vertices_per_polyhedron))
        push!(max_vertices_per_polyhedron, maximum(vertices_per_polyhedron))
        push!(mean_facets_per_polyhedron, mean(facets_per_polyhedron))
        push!(max_facets_per_polyhedron, maximum(facets_per_polyhedron))
        push!(mean_rays_per_polyhedron, mean(rays_per_polyhedron))
    end
    region_stats = DataFrame(
        Seed = fill(seed, length(epochs)),
        Epoch = epochs,
        Step = steps,
        Mean_Finite_Region_Volume = mean_finite_volumes,
        Median_Finite_Region_Volume = median_finite_volumes,
        Finite_Region_Count = finite_region_counts,
        Unbounded_Region_Count = unbounded_region_counts,
        Total_Region_Count = total_region_counts,
    )
    CSV.write(joinpath(run_path, "whole_plane_region_stats.csv"), region_stats)
    polyhedra_stats = DataFrame(
        Seed = fill(seed, length(epochs)),
        Epoch = epochs,
        Step = steps,
        Linear_Region_Count = total_region_counts,
        Bounded_Region_Count = finite_region_counts,
        Unbounded_Region_Count = unbounded_region_counts,
        Polyhedron_Count = polyhedron_counts,
        Mean_Vertices_Per_Region = mean_vertices_per_region,
        Max_Vertices_Per_Region = max_vertices_per_region,
        Mean_Vertices_Per_Constituent_Polyhedron = mean_vertices_per_polyhedron,
        Max_Vertices_Per_Constituent_Polyhedron = max_vertices_per_polyhedron,
        Mean_Facets_Per_Constituent_Polyhedron = mean_facets_per_polyhedron,
        Max_Facets_Per_Constituent_Polyhedron = max_facets_per_polyhedron,
        Mean_Rays_Per_Constituent_Polyhedron = mean_rays_per_polyhedron,
    )
    CSV.write(joinpath(run_path, "polyhedra_stats.csv"), polyhedra_stats)
    return metrics, final, monomials, region_stats, hoffman, polyhedra_stats
end

_sample_std(values) = length(values) == 1 ? 0.0 : std(values)

function summarize_training(metrics)
    return combine(
        groupby(metrics, [:Epoch, :Step]),
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
    epochs = unique(final_metrics.Epoch)
    length(epochs) == 1 || throw(ArgumentError("final metrics use different epoch counts"))
    steps = unique(final_metrics.Step)
    length(steps) == 1 || throw(ArgumentError("final metrics use different step counts"))
    return DataFrame(
        Runs = [nrow(final_metrics)],
        Epoch = [only(epochs)],
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
        groupby(monomials, [:Epoch, :Step]),
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
        groupby(volumes, [:Epoch, :Step]),
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

function summarize_hoffman(hoffman)
    return combine(
        groupby(hoffman, [:Epoch, :Step]),
        :Hoffman_Constant => mean => :Hoffman_Constant_Mean,
        :Hoffman_Constant => _sample_std => :Hoffman_Constant_Std,
    )
end

function summarize_polyhedra(polyhedra)
    measurements = filter(propertynames(polyhedra)) do name
        name != :Seed && name != :Epoch && name != :Step
    end
    transformations = Pair[]
    for measurement in measurements
        push!(transformations, measurement => mean => Symbol(measurement, "_Mean"))
        push!(transformations, measurement => _sample_std => Symbol(measurement, "_Std"))
    end
    return combine(groupby(polyhedra, [:Epoch, :Step]), transformations...)
end

function export_to_csvs(output_root)
    metric_frames = DataFrame[]
    final_frames = DataFrame[]
    monomial_frames = DataFrame[]
    region_frames = DataFrame[]
    hoffman_frames = DataFrame[]
    polyhedra_frames = DataFrame[]
    for run_path in run_directories(output_root)
        println("Exporting $(basename(run_path))")
        metrics, final, monomials, regions, hoffman, polyhedra =
            export_run_to_csvs(run_path)
        push!(metric_frames, metrics)
        push!(final_frames, final)
        push!(monomial_frames, monomials)
        push!(region_frames, regions)
        push!(hoffman_frames, hoffman)
        push!(polyhedra_frames, polyhedra)
    end

    if length(metric_frames) > 1
        all_metrics = vcat(metric_frames...)
        all_finals = vcat(final_frames...)
        all_monomials = vcat(monomial_frames...)
        all_regions = vcat(region_frames...)
        all_hoffman = vcat(hoffman_frames...)
        all_polyhedra = vcat(polyhedra_frames...)
        CSV.write(joinpath(output_root, "all_training_metrics.csv"), all_metrics)
        CSV.write(joinpath(output_root, "all_final_metrics.csv"), all_finals)
        CSV.write(joinpath(output_root, "all_monomial_counts.csv"), all_monomials)
        CSV.write(joinpath(output_root, "all_whole_plane_region_stats.csv"), all_regions)
        CSV.write(joinpath(output_root, "all_hoffman_constants.csv"), all_hoffman)
        CSV.write(joinpath(output_root, "all_polyhedra_stats.csv"), all_polyhedra)
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
        CSV.write(
            joinpath(output_root, "hoffman_constants_summary.csv"),
            summarize_hoffman(all_hoffman),
        )
        CSV.write(
            joinpath(output_root, "polyhedra_stats_summary.csv"),
            summarize_polyhedra(all_polyhedra),
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
