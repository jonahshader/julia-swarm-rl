include("nn.jl")

using Raylib
using CUDA

mutable struct Creature
    position::AbstractArray
    velocity::AbstractArray
    nn::NeuralNet
    memory::AbstractArray
end

function make_creatures(n::Int, width::Float32, height::Float32, device::Symbol = :cpu)
    position = vcat(rand(Float32, n) * width, rand(Float32, n) * height)
    velocity = zeros(Float32, 2, n)
    memory = randn(Float32, 32, n)

    nn = make_net([32, 200, 200, 3], tanh, device=device)
    change_batch_size!(nn, n)

    if device != :cpu
        position = CuMatrix(Float32, position)
        velocity = CuMatrix(Float32, velocity)
    end
    Creature(position, velocity, nn)
end

function update_creatures!(creatures::Creature)
    nn_out = forward!(creatures.nn, creatures.memory)
    nn_out_x = view(nn_out, 1, :)
    nn_out_y = view(nn_out, 2, :)
    nn_out_xy = view(nn_out, 1:2, :)
    nn_out_mag = view(nn_out, 3, :)
    scalars = nn_out_mag ./ sqrt.(nn_out_x .^ 2 + nn_out_y .^ 2)
    creatures.velocity .+= nn_out_xy .* scalars
    creatures.velocity .*= 0.9
    creatures.position .+= creatures.velocity .* (1/60)
end

function render_creatures(creatures::Creature)
    x = view(creatures.position, 1, :)
    y = view(creatures.position, 2, :)
    Raylib.DrawCircle.(Int.(round.(x)), Int.(round.(y)), 5.0, Raylib.RED)
end