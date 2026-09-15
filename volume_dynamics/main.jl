include(joinpath(@__DIR__, "..", "experiment_setup.jl"))

using Distributed
using Logging
using TropicalNN

const EXPERIMENT_RUNTIME = setup_experiment!()
const REGION_MODE = TropicalNN.OscarMode()
const WORKER_IDS = tropical_workers(EXPERIMENT_RUNTIME)

include(joinpath(@__DIR__, "experiment.jl"))
using .VolumeDynamicsExperiment

function _load_volume_dynamics_on_workers!(worker_pool)
    worker_pool === nothing && return nothing
    source_path = joinpath(@__DIR__, "experiment.jl")
    for pid in Distributed.workers(worker_pool)
        remotecall_wait(pid, source_path) do path
            isdefined(Main, :VolumeDynamicsExperiment) || include(path)
            nothing
        end
    end
    return nothing
end

_load_volume_dynamics_on_workers!(WORKER_IDS)

global_logger(SimpleLogger(stderr, Logging.Error))

function main()
    config = config_from_env()
    default_output = joinpath("outputs", "volume_dynamics", dataset_name(config.generator))
    output_root = get(ENV, "VOLUME_OUTPUT_DIR", default_output)
    run_experiment(
        config;
        output_root = output_root,
        mode = REGION_MODE,
        workers = WORKER_IDS,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
