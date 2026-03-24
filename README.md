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