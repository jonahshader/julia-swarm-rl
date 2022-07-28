include("layer.jl")

mutable struct NeuralNet
    layers::Vector{Layer}
end

# length of layer_activations is one less than layers length
function make_net(layer_sizes::Vector{Int}, layer_activations::Vector{Function}; init_scale::Float32 = 1f0, device::Symbol = :cpu)
    layers = Vector{Layer}()
    for i in 2:length(layer_sizes)
        push!(layers, make_layer(layer_sizes[i-1], layer_sizes[i], layer_activations[i-1]; init_scale=init_scale, device=device))
    end
    NeuralNet(layers)
end

function make_net(layer_sizes::Vector{Int}, activation::Function; init_scale::Float32 = 1f0, device::Symbol = :cpu)
    layers = Vector{Layer}()
    for i in 2:length(layer_sizes)
        push!(layers, make_layer(layer_sizes[i-1], layer_sizes[i], activation; init_scale=init_scale, device=device))
    end
    NeuralNet(layers)
end

function forward!(net::NeuralNet, input::AbstractArray)
    forward!(net.layers[1], input)
    for i in 2:length(net.layers)
        forward!(net.layers[i], net.layers[i-1].output)
    end
    net.layers[end].output
end

function change_batch_size!(net::NeuralNet, batch_size::Int)
    for i in 1:length(net.layers)
        change_batch_size!(net.layers[i], batch_size)
    end
    nothing
end