using JLD2
using CSV
using DataFrames
using Graphs
using Statistics
using TropicalNN

function bounded_region_volumes(graph)
    region_volumes = Float64[]

    for vertex in Graphs.vertices(graph)
        # A graph vertex represents one connected linear region. Its metadata
        # stores one volume per constituent polyhedron, so these must be summed
        # before computing statistics across regions.
        component_volumes = Float64.(graph[vertex]["volume"])
        region_volume = sum(component_volumes)
        isfinite(region_volume) && push!(region_volumes, region_volume)
    end

    return region_volumes
end

function export_to_csvs(data_path="outputs/volume_dynamics")
    
    println("Loading training data...")
    # -------------------------------
    # 1. Accuracies and Loss CSV
    # -------------------------------
    training_data = JLD2.load("$data_path/training_data.jld2")["data"]
    
    df_acc = DataFrame(
        Epoch = 0:(length(training_data["train_acc"]) - 1),
        Train_Accuracy = training_data["train_acc"],
        Test_Accuracy = training_data["test_acc"],
        Loss = training_data["loss"]
    )
    
    CSV.write("$data_path/accuracies.csv", df_acc)
    println("Saved accuracies.csv")

    # -------------------------------
    # 2. Monomial Counts CSV
    # -------------------------------
    # Get the list of epochs that were actually saved as folders
    epoch_names = filter(name -> isdir(joinpath(data_path, name)) &&
                                 occursin(r"^\d+$", name), readdir(data_path))
    epochs = sort(parse.(Int, epoch_names))

    monomial_path = joinpath(data_path, "monomial_data.jld2")
    if isfile(monomial_path)
        println("Loading monomial data...")
        monomial_data = JLD2.load(monomial_path)["data"]
        num_monomial_epochs = length(monomial_data["pre"])
        length(monomial_data["post"]) == num_monomial_epochs ||
            error("Pre- and post-pruning monomial histories have different lengths")
        # main.jl computes one count per epoch directory, in sorted order, so
        # the entries line up with `epochs`; a mismatch means the file is stale.
        num_monomial_epochs == length(epochs) ||
            error("Monomial count history does not match the epoch directories")
        df_mono = DataFrame(
            Epoch = epochs,
            Pre_Pruning = monomial_data["pre"],
            Post_Pruning = monomial_data["post"]
        )
        CSV.write(joinpath(data_path, "monomial_counts.csv"), df_mono)
        println("Saved monomial_counts.csv")
    end

    # -------------------------------
    # 3. Finite Volumes Summary Stats
    # -------------------------------
    println("Calculating volume statistics per epoch...")
    
    mean_vols = Float64[]
    median_vols = Float64[]
    count_vols = Int[]
    
    for epoch in epochs
        graph = JLD2.load("$data_path/$epoch/graph.jld2")["graph"]
        finite_vols = bounded_region_volumes(graph)
        
        if isempty(finite_vols)
            push!(mean_vols, NaN)
            push!(median_vols, NaN)
            push!(count_vols, 0)
        else
            push!(mean_vols, mean(finite_vols))
            push!(median_vols, median(finite_vols))
            push!(count_vols, length(finite_vols))
        end
    end
    
    df_vols = DataFrame(
        Epoch = epochs,
        Mean_Finite_Volume = mean_vols,
        Median_Finite_Volume = median_vols,
        Finite_Volume_Count = count_vols
    )
    
    CSV.write("$data_path/finite_volumes_stats.csv", df_vols)
    println("Saved finite_volumes_stats.csv")

    # -------------------------------
    # 4. Polyhedral Subdivision Stats
    # -------------------------------
    println("Calculating polyhedra statistics per epoch...")

    df_polys = DataFrame(
        Epoch = Int[],
        Linear_Region_Count = Int[],
        Bounded_Region_Count = Int[],
        Unbounded_Region_Count = Int[],
        Polyhedron_Count = Int[],
        Mean_Vertices = Float64[],
        Max_Vertices = Int[],
        Mean_Facets = Float64[],
        Max_Facets = Int[],
        Mean_Rays = Float64[]
    )

    for epoch in epochs
        polyhedra_path = joinpath(data_path, "$epoch", "polyhedra_data.jld2")
        isfile(polyhedra_path) || continue
        # One entry per linear region, and one entry per polyhedron of the subdivision.
        data = JLD2.load(polyhedra_path)["data"]
        push!(df_polys, (
            epoch,
            length(data["polyhedra_per_region"]),
            count(data["bounded"]),
            count(!, data["bounded"]),
            sum(data["polyhedra_per_region"]),
            mean(data["vertices"]),
            maximum(data["vertices"]),
            mean(data["facets"]),
            maximum(data["facets"]),
            mean(data["rays"])
        ))
    end

    CSV.write("$data_path/polyhedra_stats.csv", df_polys)
    println("Saved polyhedra_stats.csv")

    # -------------------------------
    # 5. Hoffman Constants
    # -------------------------------
    hoffman_path = joinpath(data_path, "hoffman_data.jld2")
    if isfile(hoffman_path)
        println("Loading Hoffman constants...")
        hoffman_data = JLD2.load(hoffman_path)["data"]
        # main.jl computes one constant per epoch directory, in sorted order, so
        # the entries line up with `epochs`; a mismatch means the file is stale.
        length(hoffman_data) == length(epochs) ||
            error("Hoffman constant history does not match the epoch directories")
        df_hoffman = DataFrame(
            Epoch = epochs,
            Hoffman_Constant = hoffman_data
        )
        CSV.write(joinpath(data_path, "hoffman_constants.csv"), df_hoffman)
        println("Saved hoffman_constants.csv")
    end

    println("All CSV data exported successfully!")
end

# Run the extraction
data_path = isempty(ARGS) ? "outputs/volume_dynamics" : ARGS[1]
export_to_csvs(data_path)
