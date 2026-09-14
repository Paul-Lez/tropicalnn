using Test
using JLD2
import Random
using Statistics
using TropicalNN

include(joinpath(@__DIR__, "..", "volume_dynamics", "experiment.jl"))
using .VolumeDynamicsExperiment
include(joinpath(@__DIR__, "..", "volume_dynamics", "analyse.jl"))

@testset "volume dynamics" begin
    generator = SpiralGenerator(noise = 0.01, turns = 0.5)
    dataset = generate_dataset(
        generator,
        Random.MersenneTwister(11);
        train_size = 20,
        validation_size = 8,
        test_size = 8,
    )
    repeated_dataset = generate_dataset(
        generator,
        Random.MersenneTwister(11);
        train_size = 20,
        validation_size = 8,
        test_size = 8,
    )

    @test dataset.train.features == repeated_dataset.train.features
    @test dataset.train.labels == repeated_dataset.train.labels
    @test size(dataset.train.features) == (20, 2)
    @test count(==(0), dataset.train.labels) == 10
    @test count(==(1), dataset.train.labels) == 10
    @test vec(mean(dataset.train.features; dims = 1)) ≈ zeros(2) atol = 1e-12
    @test vec(std(dataset.train.features; dims = 1, corrected = false)) ≈ ones(2) atol = 1e-12
    @test fieldnames(BinaryDataset) == (:train, :validation, :test, :feature_mean, :feature_scale)

    mktempdir() do output_root
        config = ExperimentConfig(generator = generator, seeds = [11])
        config_data = VolumeDynamicsExperiment._config_data(config, 11)
        @test config_data["analysis_domain"] == "whole_plane"
        @test !haskey(config_data, "analysis_margin")
        @test VolumeDynamicsExperiment._expected_checkpoint_steps(
            ExperimentConfig(max_steps = 5, checkpoint_every = 2)
        ) == [0, 2, 4, 5]
        @test_nowarn VolumeDynamicsExperiment._prepare_output_root(output_root, config)
        @test_nowarn VolumeDynamicsExperiment._prepare_output_root(output_root, config)
        changed_config = ExperimentConfig(
            generator = generator,
            learning_rate = 2e-3,
            seeds = [11],
        )
        @test_throws ArgumentError VolumeDynamicsExperiment._prepare_output_root(
            output_root,
            changed_config,
        )
    end

    mktempdir() do run_path
        run_config = ExperimentConfig(
            generator = generator,
            train_size = 20,
            validation_size = 8,
            test_size = 8,
            width = 2,
            batch_size = 4,
            learning_rate = 1e-3,
            weight_decay = 1e-4,
            max_steps = 1,
            checkpoint_every = 1,
            seeds = [11],
        )
        config_data = VolumeDynamicsExperiment._config_data(run_config, 11)
        JLD2.jldsave(joinpath(run_path, "config.jld2"); config_data)
        save_dataset(joinpath(run_path, "dataset.jld2"), dataset)
        saved_dataset = JLD2.load(joinpath(run_path, "dataset.jld2"))
        @test !haskey(saved_dataset, "analysis_lower")
        @test !haskey(saved_dataset, "analysis_upper")

        model = build_model(Random.MersenneTwister(12), 2)
        @test eltype(model[1].weight) == Float64
        training_data, final_metrics = train_model!(
            model,
            dataset,
            run_path;
            batch_size = 4,
            learning_rate = 1e-3,
            weight_decay = 1e-4,
            max_steps = 1,
            checkpoint_every = 1,
            rng = Random.MersenneTwister(13),
        )
        @test training_data["step"] == [0, 1]
        @test training_data["examples_seen"] == [0, 4]
        @test !haskey(training_data, "test_accuracy")
        @test final_metrics["step"] == 1

        initial_parameters = JLD2.load(joinpath(
            run_path, "checkpoints", "00000000", "parameters.jld2"
        ))
        trained_parameters = JLD2.load(joinpath(
            run_path, "checkpoints", "00000001", "parameters.jld2"
        ))
        @test eltype(initial_parameters["weights"][1]) == Rational{BigInt}
        @test initial_parameters["weights"] != trained_parameters["weights"]
        @test eltype(model[1].weight) == Float64

        monomial_data = analyze_checkpoints(run_path; mode = HiGHSMode())
        @test monomial_data["step"] == [0, 1]
        graph = JLD2.load(joinpath(
            run_path, "checkpoints", "00000000", "graph.jld2"
        ))["graph"]
        region_volumes = graph_region_volumes(graph)
        @test !isempty(region_volumes)
        @test any(isinf, region_volumes)
        @test all(volume -> isfinite(volume) || isinf(volume), region_volumes)
        edge_data = JLD2.load(joinpath(
            run_path, "checkpoints", "00000000", "edge_data.jld2"
        ))["edge_data"]
        @test edge_data["directions"] isa Vector{Vector{Float64}}
        @test !isempty(edge_data["directions"])
        @test all(edge_data["directions"]) do direction
            length(direction) == 2 && all(isfinite, direction) &&
                isapprox(sum(abs2, direction), 1.0; atol = 1e-12)
        end
        @test edge_data["lengths"] isa Vector{Float64}
        @test all(length -> isfinite(length) && length > 0, edge_data["lengths"])
        @test !isfile(joinpath(
            run_path, "checkpoints", "00000000", "whole_space_graph.jld2"
        ))

        metrics, final, monomials, regions = export_run_to_csvs(run_path)
        @test metrics.Step == [0, 1]
        @test final.Test_Accuracy[1] == final_metrics["test_accuracy"]
        @test monomials.Step == [0, 1]
        @test regions.Step == [0, 1]
        @test regions.Finite_Region_Count .+ regions.Unbounded_Region_Count ==
            regions.Total_Region_Count
        @test all(>(0), regions.Unbounded_Region_Count)
        @test isfile(joinpath(run_path, "whole_plane_region_stats.csv"))
        @test nrow(summarize_training(vcat(metrics, metrics))) == 2
        @test nrow(summarize_final_metrics(vcat(final, final))) == 1
        @test nrow(summarize_monomials(vcat(monomials, monomials))) == 2
        @test nrow(summarize_region_volumes(vcat(regions, regions))) == 2
        @test VolumeDynamicsExperiment._completed_run_matches(run_path, run_config, 11)
    end
end
