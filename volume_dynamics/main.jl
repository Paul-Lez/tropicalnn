include(joinpath(@__DIR__, "..", "experiment_setup.jl"))

using Logging
using TropicalNN

const EXPERIMENT_RUNTIME = setup_experiment!()
const REGION_MODE = highs_mode(EXPERIMENT_RUNTIME)
const WORKER_IDS = tropical_workers(EXPERIMENT_RUNTIME)

include(joinpath(@__DIR__, "experiment.jl"))
using .VolumeDynamicsExperiment

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
