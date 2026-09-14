using Dates
using SHA
using TOML

function git_head(path)
    return readchomp(`git -C $path rev-parse HEAD`)
end

git_dirty(path) = !isempty(readchomp(`git -C $path status --porcelain`))

function main(args = ARGS)
    isempty(args) && throw(ArgumentError("provide an output TOML path"))
    output_path = abspath(first(args))
    project_root = @__DIR__
    library_root = normpath(joinpath(project_root, "..", "TropicalNN.jl"))
    manifest_path = joinpath(project_root, "Manifest.toml")
    data = Dict{String, Any}(
        "created_at_utc" => string(Dates.now(Dates.UTC)),
        "experiment_git_revision" => git_head(project_root),
        "experiment_worktree_dirty" => git_dirty(project_root),
        "library_git_revision" => git_head(library_root),
        "library_worktree_dirty" => git_dirty(library_root),
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "cpu_name" => Sys.CPU_NAME,
        "cpu_threads" => Sys.CPU_THREADS,
        "total_memory_bytes" => Sys.total_memory(),
        "operating_system" => string(Sys.KERNEL),
        "architecture" => string(Sys.ARCH),
        "command" => join(args[2:end], " "),
        "manifest_sha256" => open(manifest_path) do io
            bytes2hex(SHA.sha256(io))
        end,
        "tropicalnn_processes" => get(ENV, "TROPICALNN_PROCESSES", "not_set"),
        "highs_threads" => get(ENV, "TROPICALNN_HIGHS_THREADS", "default_8"),
    )
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        TOML.print(io, data; sorted = true)
    end
    println("Run manifest written to $output_path")
end

main()
