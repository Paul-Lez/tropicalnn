using CSV
using DataFrames
import Flux
import Random
using Test
using TropicalNN

include(joinpath(@__DIR__, "..", "paper_exports.jl"))
using .PaperExports
include(joinpath(@__DIR__, "..", "mnist", "models.jl"))
using .MNISTModels
include(joinpath(@__DIR__, "..", "effective_radius", "hoffman_tables.jl"))

@testset "paper and experiment conformance" begin
    @testset "function-level Hoffman semantics" begin
        p = Signomial(
            [0, 0, 0, -1],
            [[0, 0], [2, 0], [0, 2], [1 // 2, 1 // 100]];
            sorted = false,
        )
        q = Signomial([0], [[0 // 1, 0 // 1]]; sorted = false)
        f = RationalSignomial(p, q)
        mode = HiGHSMode(threads = 1)

        function_value = hoffman_constant(f; brute_force = true, mode = mode)
        exponent_matrices, _ = TropicalNN._linearmap_matrices(f)
        all_pairs_value = maximum(
            matrix -> hoffman_constant(matrix; brute_force = true),
            vec(TropicalNN._tilde_matrices(exponent_matrices)),
        )
        @test function_value ≈ 100 / 51
        @test all_pairs_value ≈ 100
        @test function_cell_matrix_count(f, mode) == 3

        candidate_matrix = zeros(15, 2)
        for index in 1:13
            candidate_matrix[index + 2, :] = [1.0, index]
        end
        @test brute_force_candidate_count(candidate_matrix) == 91
        @test one_tenth_sample_count(91) == 10
        @test one_tenth_sample_count(0) == 0
        zero_statistics = lower_statistics([zeros(2, 2)])
        @test zero_statistics.candidate_count == 0
        @test zero_statistics.sample_count == 0
        @test zero_statistics.sampling_fraction == 0.0

        sample = compute_table(
            (m_p = 2, m_q = 1, n = 2);
            num_samples = 1,
            rng = Random.MersenneTwister(1),
            lower_rng = Random.MersenneTwister(2),
            benchmark = :function,
            highs_threads = 1,
        )
        @test sample.Benchmark == ["function_H_pq"]
        @test sample.CoefficientPolicy == ["all_zero"]
        @test sample.LowerSamplePolicy == [LOWER_SAMPLE_POLICY]
        @test sample.LowerSamples[1] >= cld(sample.BruteForceCandidates[1], 10)
        @test sample.LowerSamples[1] <=
            cld(sample.BruteForceCandidates[1], 10) + sample.CellMatrices[1] - 1
        @test sample.LowerSamplingFraction[1] ==
            sample.LowerSamples[1] / sample.BruteForceCandidates[1]
        @test sample.MinLowerSamplesPerCellMatrix[1] >= 1
        @test sample.MaxLowerSamplesPerCellMatrix[1] >=
            sample.MinLowerSamplesPerCellMatrix[1]
        @test sample.PVZHoffman ≈ sample.BruteForceHoffman
        @test all(>=(0), sample.LowerSeconds)
        summary_input = copy(sample)
        insertcols!(
            summary_input,
            1,
            :MP => [2],
            :MQ => [1],
            :N => [2],
        )
        summary = hoffman_summary(summary_input, 1)
        @test summary.MeanBruteForceCandidates ==
            Float64.(sample.BruteForceCandidates)
        @test summary.MeanLowerSamples == Float64.(sample.LowerSamples)
        @test summary.MeanLowerSamplingFraction == sample.LowerSamplingFraction
        @test_throws ErrorException run_hoffman_tables([
            "--hoffman-lower-samples=100",
        ])
    end

    @testset "Hoffman paper aggregation" begin
        samples = DataFrame(
            Benchmark = fill("function_H_pq", 2),
            MP = fill(2, 2),
            MQ = fill(3, 2),
            N = fill(6, 2),
            Sample = [1, 2],
            LowerHoffman = [1.0, 3.0],
            LowerSeconds = [0.1, 0.2],
            BruteForceHoffman = [1.0, 1.0],
            BruteForceSeconds = [0.3, 0.5],
            PVZHoffman = [1.0, 1.0 + eps()],
            PVZSeconds = [0.01, 0.02],
            UpperHoffman = [4.0, 2.0],
            UpperSeconds = [0.001, 0.002],
        )
        raw, long, wide = hoffman_paper_tables(samples)
        @test long.MeanLowerAbsoluteError == [1.0]
        @test long.MeanBruteForceHoffman == [1.0]
        @test wide[wide.Quantity .== "Lower", :mp2_mq3_n6] == [1.0]
        @test wide[wide.Quantity .== "PVZ", :mp2_mq3_n6][1] <= eps()
        @test raw.LowerAbsoluteError == [0.0, 2.0]
    end

    @testset "MNIST model conversion keeps logits" begin
        point = Float32[0.25, -0.75]
        for spec in (
            ExperimentSpec("relu", :relu, [2]),
            ExperimentSpec("maxout", :maxout, [2]; pieces = 2),
            ExperimentSpec("relu2", :relu, [2, 2]),
        )
            Random.seed!(17)
            model = build_model(spec; input_dimension = 2, output_dimension = 2)
            hidden = point
            for layer_index in eachindex(spec.widths)
                hidden = model[layer_index](hidden)
            end
            flux_logits = model[length(spec.widths) + 1](hidden)
            tropical_network = model_to_tropical(model, spec)
            tropical_output = tropicalize(tropical_network; quicksum = true, dedup = true)
            tropical_logits = TropicalNN.TropicalNumbers.content.(
                TropicalNN.evaluate(tropical_output, Float64.(point)),
            )
            @test tropical_logits ≈ Float64.(flux_logits) atol = 1e-6
        end
    end

    @testset "archived samples reproduce paper aggregates" begin
        archive_root = joinpath(@__DIR__, "..", "outputs", "big-run")
        mnist_regions_raw = CSV.read(
            joinpath(archive_root, "mnist", "linear_regions.csv"),
            DataFrame,
        )
        mnist_metrics_raw = CSV.read(
            joinpath(archive_root, "mnist", "metrics.csv"),
            DataFrame,
        )
        width_depth_raw = CSV.read(
            joinpath(archive_root, "width_depth", "linear_regions.csv"),
            DataFrame,
        )
        _, _, mnist_regions, mnist_metrics = mnist_paper_tables(
            mnist_regions_raw,
            mnist_metrics_raw,
        )
        all_mnist_regions = PaperExports._mnist_region_summary(mnist_regions_raw)
        all_mnist_metrics = PaperExports._mnist_metrics_summary(mnist_metrics_raw)
        width_depth = width_depth_paper_table(width_depth_raw)
        @test nrow(mnist_regions_raw) == 450
        @test nrow(width_depth_raw) == 720
        @test nrow(mnist_regions) == 10
        @test nrow(mnist_metrics) == 10
        @test nrow(width_depth) == 24
        @test all(==(30), mnist_regions.NumSamples)
        @test all(==(30), mnist_metrics.NumSamples)
        @test all(==(30), width_depth.NumSamples)

        expected_mnist = Dict(
            "relu_d1_w4" => (16.0, 9.25290480867, 0.842983333333),
            "relu_d1_w5" => (32.0, 3.34846148553, 0.876013333333),
            "relu_d1_w6" => (64.0, 5.6064969283, 0.906523333333),
            "relu_d1_w7" => (128.0, 10.9527585216, 0.918376666667),
            "relu_d1_w8" => (256.0, 23.1344616681, 0.92419),
            "maxout2_d1_w4" => (16.0, 2.39149817407, 0.913856666667),
            "maxout2_d1_w5" => (32.0, 3.40245405243, 0.92916),
            "maxout2_d1_w6" => (64.0, 6.77039544057, 0.938306666667),
            "maxout2_d1_w7" => (128.0, 11.7034347976, 0.94128),
            "maxout2_d1_w8" => (256.0, 24.611718978, 0.94469),
            "relu_d2_w4" => (84.1333333333, 32.6503823318, 0.82673),
            "relu_d2_w5" => (332.766666667, 18.5419997416, 0.885036666667),
            "relu_d2_w6" => (1242.56666667, 59.2671736752, 0.904833333333),
            "relu_d2_w7" => (4684.53333333, 287.371788369, 0.918586666667),
            "relu_d2_w8" => (16532.6, 1369.76371057, 0.92345),
        )
        for row in eachrow(all_mnist_regions)
            expected_regions, expected_time, _ = expected_mnist[row.Model]
            @test row.NumRegions ≈ expected_regions rtol = 1e-10
            @test row.TimeSeconds ≈ expected_time rtol = 1e-10
        end
        for row in eachrow(all_mnist_metrics)
            _, _, expected_accuracy = expected_mnist[row.Model]
            @test row.TestAccuracy ≈ expected_accuracy rtol = 1e-10
        end

        expected_width_regions = Dict(
            ("relu", "depth", 1) => 56.0,
            ("maxout", "depth", 1) => 56.0,
            ("relu", "depth", 2) => 216.133333333,
            ("maxout", "depth", 2) => 215.466666667,
            ("relu", "depth", 3) => 453.3,
            ("maxout", "depth", 3) => 454.566666667,
            ("relu", "depth", 4) => 845.666666667,
            ("maxout", "depth", 4) => 884.166666667,
            ("relu", "depth", 5) => 1268.4,
            ("maxout", "depth", 5) => 1232.66666667,
            ("relu", "depth", 6) => 1689.03333333,
            ("maxout", "depth", 6) => 2069.93333333,
            ("relu", "width", 10) => 56.0,
            ("maxout", "width", 10) => 56.0,
            ("relu", "width", 20) => 211.0,
            ("maxout", "width", 20) => 211.0,
            ("relu", "width", 30) => 466.0,
            ("maxout", "width", 30) => 466.0,
            ("relu", "width", 40) => 820.966666667,
            ("maxout", "width", 40) => 820.966666667,
            ("relu", "width", 50) => 1276.0,
            ("maxout", "width", 50) => 1276.0,
            ("relu", "width", 60) => 1830.93333333,
            ("maxout", "width", 60) => 1830.9,
        )
        expected_width_times = Dict(
            ("relu", "depth", 1) => 8.1333180128,
            ("maxout", "depth", 1) => 0.8740586144,
            ("relu", "depth", 2) => 3.92033948633,
            ("maxout", "depth", 2) => 2.70651895977,
            ("relu", "depth", 3) => 5.91520386783,
            ("maxout", "depth", 3) => 5.44711922493,
            ("relu", "depth", 4) => 11.0828855432,
            ("maxout", "depth", 4) => 11.5179811693,
            ("relu", "depth", 5) => 20.8091262527,
            ("maxout", "depth", 5) => 19.9332085134,
            ("relu", "depth", 6) => 33.8973991638,
            ("maxout", "depth", 6) => 40.3042490147,
            ("relu", "width", 10) => 1.6343526842,
            ("maxout", "width", 10) => 1.0057126296,
            ("relu", "width", 20) => 6.33594805203,
            ("maxout", "width", 20) => 4.53656375847,
            ("relu", "width", 30) => 15.1885437959,
            ("maxout", "width", 30) => 12.9376676549,
            ("relu", "width", 40) => 30.9304190623,
            ("maxout", "width", 40) => 28.3603373247,
            ("relu", "width", 50) => 56.5333282821,
            ("maxout", "width", 50) => 54.3251210371,
            ("relu", "width", 60) => 89.2741560655,
            ("maxout", "width", 60) => 85.3199867165,
        )
        for row in eachrow(width_depth)
            key_value = row.Sweep == "depth" ? row.HiddenLayers : row.Width
            key = (row.Network, row.Sweep, key_value)
            @test row.NumRegions ≈ expected_width_regions[key] rtol = 1e-10
            @test row.TimeSeconds ≈ expected_width_times[key] rtol = 1e-10
        end

        spec_indices = Dict(spec.id => index for (index, spec) in enumerate(experiment_specs()))
        @test all(eachrow(mnist_metrics_raw)) do row
            row.Seed == 20260824 + spec_indices[row.Model] +
                (row.Sample - 1) * length(spec_indices)
        end
        architecture_indices = Dict{Tuple{String, String}, Int}()
        index = 0
        for depth in 1:6
            index += 1
            architecture_indices[("depth", join(vcat(2, fill(10, depth), 1), ":"))] = index
        end
        for width in 10:10:60
            index += 1
            architecture_indices[("width", "2:$width:1")] = index
        end
        @test all(eachrow(width_depth_raw)) do row
            network_index = row.Network == "relu" ? 1 : 2
            architecture_index = architecture_indices[(row.Sweep, row.Architecture)]
            row.Seed == 20260824 + 1_000 * architecture_index +
                100 * network_index + row.Trial
        end
    end
end
