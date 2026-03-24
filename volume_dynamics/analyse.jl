using JLD2
using CSV
using DataFrames
using Statistics

function export_to_csvs()
    data_path = "outputs/volume_dynamics"
    
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
    println("Loading monomial data...")
    monomial_data = JLD2.load("$data_path/monomial_data.jld2")["data"]
    
    # Get the list of epochs that were actually saved as folders
    epochs = sort(parse.(Int, filter(x -> !occursin(".", x), readdir(data_path))))
    
    df_mono = DataFrame(
        Epoch = epochs,
        Pre_Pruning = monomial_data["pre"],
        Post_Pruning = monomial_data["post"]
    )
    
    CSV.write("$data_path/monomial_counts.csv", df_mono)
    println("Saved monomial_counts.csv")

    # -------------------------------
    # 3. Finite Volumes Summary Stats
    # -------------------------------
    println("Calculating volume statistics per epoch...")
    
    mean_vols = Float64[]
    median_vols = Float64[]
    count_vols = Int[]
    
    for epoch in epochs
        edge_data = JLD2.load("$data_path/$epoch/edge_data.jld2")["data"]
        
        # Convert lengths to Float64 (handles Rational{BigInt} parsing safely)
        lengths = Float64.(edge_data["lengths"])
        
        # Filter out infinite lengths (unbounded rays in the tropical curve)
        finite_vols = filter(isfinite, lengths)
        
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
    println("All data exported successfully!")
end

# Run the extraction
export_to_csvs()