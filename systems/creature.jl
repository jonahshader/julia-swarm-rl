include("nn.jl")

mutable struct Creature
    position::AbstractArray
    velocity::AbstractArray
    nn::NeuralNet
    memory::AbstractArray
    has_tile::Bool
end