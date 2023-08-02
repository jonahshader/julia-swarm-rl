using CUDA
using LinearAlgebra

mutable struct Layer
    weights::AbstractMatrix
    bias::AbstractVector
    output::AbstractArray
    activation::Union{Function, Nothing}
end

function make_layer(input_size::Int, output_size::Int, activation::Union{Function, Nothing}; init_scale::Float32, device::Symbol = :cpu)
    if device == :cpu
        weights = randn(Float32, output_size, input_size) * init_scale
        biases = randn(Float32, output_size) * init_scale
        output_temp = zeros(Float32, output_size)
        Layer(weights, biases, output_temp, activation)
    else
        weights = CuMatrix(randn(Float32, output_size, input_size) * init_scale)
        biases = CuVector(randn(Float32, output_size) * init_scale)
        output_temp = CuVector(zeros(Float32, output_size))
        Layer(weights, biases, output_temp, activation)
    end
end

function forward!(layer::Layer, input::AbstractVector)
    copyto!(layer.output, layer.bias)
    mul!(layer.output, layer.weights, input, 1, 1)
    if !isnothing(layer.activation)
        layer.output .= layer.output .|> layer.activation
    end
    layer.output
end

# TODO: output size could be wrong
function forward!(layer::Layer, input::AbstractMatrix)
    mul!(layer.output, layer.weights, input)
    layer.output .+= layer.bias
    if !isnothing(layer.activation)
        layer.output .= layer.output .|> layer.activation
    end
end

function change_batch_size!(layer::Layer, batch_size::Int)
    layer.output = similar(layer.output, size(layer.output)[1], batch_size)
    nothing
end