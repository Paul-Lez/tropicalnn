### Setup

Works with Julia 1.11+.
Start by instantiating the project dependecies using the following commands.

```bash
julia --project=.
```

```julia
using Pkg
Pkg.instantiate()
```

From the root directory, run the `main.jl` in the experiment directory, and then `analyse.jl` if present.

### Volume dynamics

The volume-dynamics experiment trains in `Float64`, saves exact-rational
checkpoints, and performs its polyhedral analysis only after each checkpoint has
been converted to rational arithmetic. Its default dataset is a reproducible
two-arm spiral generated with `MLUtils`.

```bash
julia --project=. volume_dynamics/main.jl
julia --project=. volume_dynamics/analyse.jl
```

Runs are separated by seed under `outputs/volume_dynamics/<dataset>/seed_<seed>`.
The default uses three seeds and records training/validation metrics at fixed
optimizer-step intervals. The test set is evaluated only once per seed.

The main settings can be changed without editing the experiment:

```bash
VOLUME_DATASET=linear_bands VOLUME_SEEDS=1,2,3 \
VOLUME_MAX_STEPS=2000 VOLUME_CHECKPOINT_EVERY=50 \
julia --project=. volume_dynamics/main.jl
```

Supported dataset names are `spiral` and `linear_bands`. Spiral geometry is
controlled by `VOLUME_SPIRAL_NOISE`, `VOLUME_SPIRAL_TURNS`, and
`VOLUME_SPIRAL_RADIUS`. To add a generator in code, subtype
`AbstractBinaryDataGenerator` and implement `dataset_name` and `generate_split`;
the normalization, splitting, plotting, and training pipeline will then be reused.
