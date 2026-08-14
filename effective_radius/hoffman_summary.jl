using CSV
using DataFrames
using Statistics

const HOFFMAN_SUMMARY_QUANTITIES = [
    "Lower",
    "LowerTime",
    "BruteForce",
    "BruteForceTime",
    "PVZ",
    "PVZTime",
    "Upper",
    "UpperTime",
]

function hoffman_configuration_name(config)
    return "mp$(config.m_p)_mq$(config.m_q)_n$(config.n)"
end

function summarize_hoffman_table(table)
    brute_force = table.BruteForceHoffman
    # Value rows are mean absolute errors relative to the brute-force baseline.
    return [
        mean(abs.(table.LowerHoffman .- brute_force)),
        mean(table.LowerMeanSeconds),
        0.0,
        mean(table.BruteForceMeanSeconds),
        mean(abs.(table.PVZHoffman .- brute_force)),
        mean(table.PVZMeanSeconds),
        mean(abs.(table.UpperHoffman .- brute_force)),
        mean(table.UpperMeanSeconds),
    ]
end

function hoffman_summary(configurations, tables)
    length(configurations) == length(tables) ||
        throw(ArgumentError("Expected one Hoffman table per configuration"))

    summary = DataFrame(Quantity = HOFFMAN_SUMMARY_QUANTITIES)
    for (config, table) in zip(configurations, tables)
        column = Symbol(hoffman_configuration_name(config))
        column in propertynames(summary) &&
            throw(ArgumentError("Duplicate Hoffman configuration: $column"))
        summary[!, column] = summarize_hoffman_table(table)
    end
    return summary
end

function write_hoffman_summary(output_dir, configurations, tables)
    output_path = joinpath(output_dir, "hoffman_summary.csv")
    CSV.write(output_path, hoffman_summary(configurations, tables))
    return output_path
end
