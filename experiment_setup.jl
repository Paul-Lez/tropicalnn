using Distributed

const DEFAULT_HIGHS_THREADS = 8

Base.@kwdef struct ExperimentRuntime
    workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
    highs_threads::Int = DEFAULT_HIGHS_THREADS
end

function _active_worker_ids()
    ids = Int[pid for pid in workers() if pid != myid()]
    return ids
end

function _parse_optional_int(args, names, env_name)
    for (idx, arg) in pairs(args)
        for name in names
            if arg == name
                idx < length(args) ||
                    throw(ArgumentError("Expected an integer after $name"))
                return parse(Int, args[idx + 1])
            end

            prefix = string(name, "=")
            if startswith(arg, prefix)
                return parse(Int, arg[(lastindex(prefix) + 1):end])
            end
        end
    end

    value = get(ENV, env_name, nothing)
    return value === nothing || isempty(value) ? nothing : parse(Int, value)
end

function _parse_int(args, names, env_name, default)
    value = _parse_optional_int(args, names, env_name)
    return value === nothing ? default : value
end

function _configure_highs_environment!(threads::Int)
    threads >= 1 || throw(ArgumentError("HiGHS thread count must be positive, got $threads"))
    ENV["OMP_NUM_THREADS"] = string(threads)
end

function _configure_worker_environment!(worker_ids, threads::Int)
    for pid in worker_ids
        remotecall_wait(pid, threads) do worker_threads
            ENV["OMP_NUM_THREADS"] = string(worker_threads)
        end
    end
end

function _load_tropicalnn_on_workers!(worker_ids)
    for pid in worker_ids
        remotecall_wait(pid) do
            Base.eval(Main, :(using TropicalNN))
        end
    end
end

function setup_experiment!(;
        args = ARGS,
        project_root = @__DIR__,
        default_highs_threads = DEFAULT_HIGHS_THREADS,
)
    requested_workers = _parse_optional_int(
        args,
        ["--processes", "--workers", "-p"],
        "TROPICALNN_PROCESSES",
    )
    highs_threads = _parse_int(
        args,
        ["--highs-threads"],
        "TROPICALNN_HIGHS_THREADS",
        default_highs_threads,
    )

    _configure_highs_environment!(highs_threads)

    existing_workers = _active_worker_ids()
    if requested_workers !== nothing
        requested_workers >= 0 ||
            throw(ArgumentError("Worker process count must be nonnegative, got $requested_workers"))
        workers_to_add = max(0, requested_workers - length(existing_workers))
        if workers_to_add > 0
            addprocs(workers_to_add; exeflags = ["--project=$(abspath(project_root))"])
        end
    end

    worker_ids = _active_worker_ids()
    if requested_workers !== nothing
        worker_ids = worker_ids[1:min(requested_workers, length(worker_ids))]
    end

    _configure_worker_environment!(worker_ids, highs_threads)
    _load_tropicalnn_on_workers!(worker_ids)
    return ExperimentRuntime(
        workers = isempty(worker_ids) ? nothing : Distributed.WorkerPool(worker_ids),
        highs_threads = highs_threads,
    )
end

function tropical_workers(runtime::ExperimentRuntime)
    return runtime.workers
end

function highs_mode(runtime::ExperimentRuntime; kwargs...)
    return TropicalNN.HiGHSMode(; threads = runtime.highs_threads, kwargs...)
end
