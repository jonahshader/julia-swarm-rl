include("nn.jl")



using CUDA
using Raylib

mutable struct Agent
    position::AbstractArray
    velocity::AbstractArray
    nn::NeuralNet
    scalars_temp::AbstractArray
    memory::AbstractArray
    vision::AbstractArray
    input::AbstractArray
end

function make_agents(n::Int, width::Float32, height::Float32, hidden_layer_sizes::Vector{Int}, memory_size::Int, vision_size::Int; device::Symbol = :cpu, activation::Function = tanh, rand_memory::Bool = false)
    # 4 channels per ray, all of memory is read in
    input_size = vision_size * 4 + memory_size
    # x y mag, all of memory has value and proportion
    output_size = 3 + memory_size * 2
    position = vcat(rand(Float32, 1, n) * width, rand(Float32, 1, n) * height)
    velocity = zeros(Float32, 2, n)
    if rand_memory
        memory = randn(Float32, memory_size, n)
    else
        memory = zeros(Float32, memory_size, n)
    end
    vision = zeros(Float32, vision_size * 4, n)
    input = zeros(Float32, input_size, n)

    nn = make_net([input_size, hidden_layer_sizes..., output_size], activation, device=device)
    change_batch_size!(nn, n)

    scalars_temp = zeros(Float32, n)

    if device != :cpu
        position = cu(position)
        velocity = cu(velocity)
        memory = cu(memory)
        vision = cu(vision)
        input = cu(input)
        scalars_temp = cu(scalars_temp)
    end
    Agent(position, velocity, nn, scalars_temp, memory, vision, input)
end


# TODO: where i left off: rewrite this to use view on memory, and i guess make placeholder views for vision
function update_agents!(agents::Agent)
    nn_input_vision = view(agents.input, 1:size(agents.vision)[1], :)
    nn_input_memory = view(agents.input, size(agents.vision)[1]+1:size(agents.vision)[1] + size(agents.memory)[1], :)
    nn_input_vision .= agents.vision
    nn_input_memory .= agents.memory

    nn_out = forward!(agents.nn, agents.input)
    nn_out_x = view(nn_out, 1, :)
    nn_out_y = view(nn_out, 2, :)
    nn_out_mag = view(nn_out, 3, :)
    vel_x = view(agents.velocity, 1, :)
    vel_y = view(agents.velocity, 2, :)
    nn_out_mem_val = view(nn_out, 3:size(agents.memory)[1]+2, :)
    nn_out_mem_prop = view(nn_out, size(agents.memory)[1]+3:2+size(agents.memory)[1]*2, :)
    agents.memory .= (nn_out_mem_val .* (nn_out_mem_prop .* .5 .+ .5)) .+ agents.memory .* (nn_out_mem_prop .* -.5 .+ .5)
    agents.scalars_temp .= nn_out_mag ./ sqrt.(nn_out_x .^ 2 + nn_out_y .^ 2)
    # agents.velocity .+= nn_out_xy .* agents.scalars_temp
    vel_x .+= nn_out_x .* agents.scalars_temp
    vel_y .+= nn_out_y .* agents.scalars_temp
    agents.velocity .*= 0.9f0
    agents.position .+= agents.velocity .* Float32(1/60)
    nothing
end



function render_agents(agents::Agent)
    pos = agents.position

    if typeof(pos) <: CuArray
        pos = Matrix(pos)
    end
    x = view(pos, 1, :)
    y = view(pos, 2, :)
    c = Raylib.RayColor(0/255, 255/255, 255/255, 255/255)
    function draw_circ(x, y)
        Raylib.DrawCircle(x, y, 5.0, c)
    end
    draw_circ.(Int.(round.(x)), Int.(round.(y)))
end