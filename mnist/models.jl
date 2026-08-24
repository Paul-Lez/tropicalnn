module MNISTModels

import Flux
using TropicalNN

export ExperimentSpec,
       build_model,
       experiment_specs,
       model_path,
       model_to_tropical

const INPUT_DIMENSION = 28^2
const OUTPUT_DIMENSION = 10
const MAXOUT_PIECES = 2

struct ExperimentSpec
    id::String
    activation::Symbol
    widths::Vector{Int}
    pieces::Int
end

function ExperimentSpec(id, activation, widths; pieces = 1)
    activation in (:relu, :maxout) ||
        throw(ArgumentError("activation must be :relu or :maxout"))
    !isempty(widths) || throw(ArgumentError("at least one hidden width is required"))
    all(>(0), widths) || throw(ArgumentError("hidden widths must be positive"))
    pieces > 0 || throw(ArgumentError("the number of maxout pieces must be positive"))
    return ExperimentSpec(String(id), activation, collect(Int, widths), Int(pieces))
end

struct MaxoutDense{D}
    affine::D
    width::Int
    pieces::Int
end

Flux.@layer MaxoutDense trainable = (affine,)

function _maxout_dense(input_dimension::Integer, width::Integer, pieces::Integer)
    affine = Flux.Dense(input_dimension => width * pieces)
    return MaxoutDense(affine, Int(width), Int(pieces))
end

function (layer::MaxoutDense)(input::AbstractArray)
    affine_output = layer.affine(input)
    output_shape = (layer.pieces, layer.width, size(affine_output)[2:end]...)
    grouped_output = reshape(affine_output, output_shape)
    return dropdims(maximum(grouped_output; dims = 1); dims = 1)
end

function experiment_specs()
    specs = ExperimentSpec[]
    for width in 4:8
        push!(specs, ExperimentSpec("relu_d1_w$width", :relu, [width]))
    end
    for width in 4:8
        push!(specs, ExperimentSpec(
            "maxout$(MAXOUT_PIECES)_d1_w$width",
            :maxout,
            [width];
            pieces = MAXOUT_PIECES,
        ))
    end
    for width in 4:8
        push!(specs, ExperimentSpec("relu_d2_w$width", :relu, [width, width]))
    end
    return specs
end

function build_model(
        spec::ExperimentSpec;
        input_dimension::Integer = INPUT_DIMENSION,
        output_dimension::Integer = OUTPUT_DIMENSION,
)
    layers = Any[]
    previous_width = input_dimension
    for width in spec.widths
        if spec.activation == :relu
            push!(layers, Flux.Dense(previous_width => width, Flux.relu))
        else
            push!(layers, _maxout_dense(previous_width, width, spec.pieces))
        end
        previous_width = width
    end
    push!(layers, Flux.Dense(previous_width => output_dimension))
    push!(layers, Flux.softmax)
    return Flux.Chain(layers...)
end

function _float64_affine(layer::Flux.Dense)
    return AffineLayer(Float64.(layer.weight), Float64.(layer.bias))
end

function model_to_tropical(model, spec::ExperimentSpec)
    layers = AbstractNeuralNetworkLayer{Float64}[]
    for hidden_index in eachindex(spec.widths)
        hidden_layer = model[hidden_index]
        width = spec.widths[hidden_index]
        if spec.activation == :relu
            push!(layers, _float64_affine(hidden_layer))
            push!(layers, ActivationLayer(relu(Float64), width))
        else
            push!(layers, _float64_affine(hidden_layer.affine))
            push!(layers, ActivationLayer(maxout(Float64, spec.pieces), width))
        end
    end

    # The final softmax is omitted: it is not piecewise linear and does not
    # change the partition induced by the preceding logits network.
    push!(layers, _float64_affine(model[length(spec.widths) + 1]))
    network = NeuralNetwork(layers)
    @assert network isa NeuralNetwork{Float64}
    return network
end

model_path(output_dir, spec::ExperimentSpec) =
    joinpath(output_dir, "models", "$(spec.id).jld2")

end
