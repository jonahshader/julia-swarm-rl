include("nn.jl")

using Raylib
using CUDA

mutable struct Creature
    position::AbstractArray
    velocity::AbstractArray
    nn::NeuralNet
    scalars_temp::AbstractArray
    memory::AbstractArray
end

function make_creatures(n::Int, width::Float32, height::Float32; hidden_layer_sizes::Vector{Int}, memory_size::Int, vision_size::Int; device::Symbol = :cpu, activation::Function = tanh)
    # 4 channels per ray, all of memory is read in
    input_size = vision_size * 4 + memory_size
    # x y mag, all of memory has value and proportion
    output_size = 3 + memory_size * 2
    position = vcat(rand(Float32, 1, n) * width, rand(Float32, 1, n) * height)
    velocity = zeros(Float32, 2, n)
    memory = randn(Float32, 32, n)

    nn = make_net([input_size, hidden_layer_sizes..., output_size], activation, device=device)
    change_batch_size!(nn, n)

    scalars_temp = zeros(Float32, n)

    if device != :cpu
        position = CuMatrix(Float32, position)
        velocity = CuMatrix(Float32, velocity)
        memory = CuMatrix(Float32, memory)
        scalars_temp = CuMatrix(Float32, scalars_temp)
    end
    Creature(position, velocity, nn, scalars_temp, memory)
end


# TODO: where i left off: rewrite this to use view on memory, and i guess make placeholder views for vision
function update_creatures!(creatures::Creature)
    nn_out = forward!(creatures.nn, creatures.memory)
    nn_out_x = view(nn_out, 1, :)
    nn_out_y = view(nn_out, 2, :)
    nn_out_xy = view(nn_out, 1:2, :)
    nn_out_mag = view(nn_out, 3, :)
    vel_x = view(creatures.velocity, 1, :)
    vel_y = view(creatures.velocity, 2, :)
    creatures.scalars_temp .= nn_out_mag ./ sqrt.(nn_out_x .^ 2 + nn_out_y .^ 2)
    # creatures.velocity .+= nn_out_xy .* creatures.scalars_temp
    vel_x .+= nn_out_x .* creatures.scalars_temp
    vel_y .+= nn_out_y .* creatures.scalars_temp
    creatures.velocity .*= 0.9f0
    creatures.position .+= creatures.velocity .* Float32(1/60)
    nothing
end

function render_creatures(creatures::Creature)
    x = view(creatures.position, 1, :)
    y = view(creatures.position, 2, :)
    Raylib.DrawCircle.(Int.(round.(x)), Int.(round.(y)), 5.0, Raylib.RED)
end