include("nn.jl")
include("math.jl")



using CUDA
using Raylib

mutable struct Agent
    position::AbstractArray
    velocity::AbstractArray
    nn::NeuralNet
    scalars_temp::AbstractArray
    memory::AbstractArray
    vision::AbstractArray
    vision_angle::AbstractArray
    vision_reduction::Int
    input::AbstractArray
end

function make_agents(n::Int, width::Float32, height::Float32, hidden_layer_sizes::Vector{Int}, memory_size::Int, vision_size::Int; vision_channels::Int = 4, vision_reduction::Int = 4, device::Symbol = :cpu, activation::Function = tanh, rand_memory::Bool = false)
    # vision, all of memory is read in
    input_size = ((vision_size * vision_channels)/vision_reduction) + memory_size
    # x y mag, all of memory has value and proportion
    output_size = 3 + memory_size * 2
    position = vcat(rand(Float32, 1, n) * width, rand(Float32, 1, n) * height)
    velocity = zeros(Float32, 2, n)
    if rand_memory
        memory = randn(Float32, memory_size, n)
    else
        memory = zeros(Float32, memory_size, n)
    end
    vision = zeros(Float32, vision_size * vision_channels, n)
    vision_angle = vcat([[cos(i*2pi/vision_size) sin(i*2pi/vision_size)] for i in 0:vision_size-1]...)
    input = zeros(Float32, input_size, n)

    nn = make_net([input_size, hidden_layer_sizes..., output_size], activation, device=device)
    change_batch_size!(nn, n)

    scalars_temp = zeros(Float32, n)

    if device != :cpu
        position = cu(position)
        velocity = cu(velocity)
        memory = cu(memory)
        vision = cu(vision)
        vision_angle = cu(vision_angle)
        input = cu(input)
        scalars_temp = cu(scalars_temp)
    end
    Agent(position, velocity, nn, scalars_temp, memory, vision, vision_angle, vision_reduction, input)
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

function update_vision_single(vis_store_size::Int, vision_angle::AbstractArray, pos::AbstractArray, circles::AbstractArray, hit_count_threshold::Int)
    vis_size = size(vision_angle)[1]

    # create ax ay bx by matrix. a is just the center and b is extended out by vision angles * 1024
    lines_vec = hcat(repeat(pos, inner = (size, 1)), repeat(pos, inner = (size, 1)) .+ repeat(vision_angle, size, 1) .* 1024)
    line_circle_collisions_vec = circle_line_intersecting(
    repeat(lines_vec[:, 1], 1, c), 
    repeat(lines_vec[:, 2], 1, c), 
    repeat(lines_vec[:, 3], 1, c), 
    repeat(lines_vec[:, 4], 1, c), 
    repeat(circles[:, 1], 1, l)', 
    repeat(circles[:, 2], 1, l)', 
    repeat(circles[:, 3], 1, l)')
    reduction_ratio = vis_size / vis_store_size

    hits = sum(line_circle_collisions_vec, dims = 2) .> hit_count_threshold
    return [max(hits[i:i+reduction_ratio]) for i in 1:reduction_ratio:vis_size]
end

# TODO: finish this
# need to add radius to agents
# need to handle device for inputs, not just update_vision_single output
function update_vision!(agents::Agent, other_circles::Vector{AbstractArray})
    views = [view(agents.vision, :, i) for i in 1:size(agents.vision)[2]]
    if typeof(agents.position) <: CuArray
        views[1] .= cu(Float32.(update_vision_single(size(agents.vision)[1], agents.vision_angle, agents.position, hcat(agents.position, ones(size(agents.position)[1])), 1)))
        for (i, circles) in enumerate(other_circles)
            views[i+1] .= cu(Float32.(update_vision_single(size(agents.vision)[1], agents.vision_angle, agents.position, circles, 0)))
        end
    else

    end
    

end


function render_agents(agents::Agent)
    pos = agents.position
    vis_angle = agents.vision_angle

    if typeof(pos) <: CuArray
        pos = Matrix(pos)
        vis_angle = Matrix(vis_angle)
    end

    x = view(pos, 1, :)
    y = view(pos, 2, :)
    c = Raylib.RayColor(0/255, 255/255, 255/255, 255/255)
    function draw_circ(x, y)
        Raylib.DrawCircle(x, y, 5.0, c)
    end
    draw_circ.(Int.(round.(x)), Int.(round.(y)))
end