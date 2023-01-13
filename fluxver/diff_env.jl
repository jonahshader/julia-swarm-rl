using Flux
using CUDA

include("agent.jl")
include("../systems/math.jl")

struct BaseEnv
    agent::Agent
end

function BaseEnv(;mem_size::Integer = 32, batch_size::Integer = 64)
    other_input_size::Integer = 4

    a = Agent(vision_size, vision_features, mem_size, other_input_size, batch_size, vision_downscale)
    BaseEnv(a)
end

Flux.@functor BaseEnv
Flux.trainable(m::BaseEnv) = (m.agent,)

function (m::BaseEnv)()
    # run iteration
    device = if m.agent.recur.state[1] |> typeof <: CuArray
        gpu
    else
        cpu
    end


    n = size(m.agent.recur.state[1])[end]
    # temp_other_input = randn(Float32, 4, n) |> device
    other_input = vcat(m.agent.recur.state[2], m.agent.recur.state[1][1:1, :], m.agent.recur.state[1][2:2, :])
    # other_input = vcat(m.agent.recur.state[2], cos.(m.agent.recur.state[1] ./ 32), sin.(m.agent.recur.state[1] ./ 32))

    vision = reshape(update_vision_soft(m.vision_angle |> cpu, m.agent.recur.state[1] |> cpu)[:, :, 1], :, 1, n) |> device
    
    # run the agent with inputs, return its output
    m.agent((vision, other_input))
end

function update_vision_soft(vision_angle::AbstractArray, pos::AbstractArray, circles::AbstractArray)
    vis_size = size(vision_angle)[2]
    pos_size = size(pos)[end]

    a = repeat(pos, outer = (1, vis_size))
    b = a .+ repeat(vision_angle, inner = (1, pos_size)) .* 4
    
    l = size(a)[end]
    c = size(circles)[end]
    
    d = line_sdf(
        repeat(a, outer = (1, c)),
        repeat(b, outer = (1, c)),
        repeat(circles, inner = (1, l))
    )
    return reshape(smoothstep(1 .- d), pos_size, vis_size, :)
end

function update_vision_soft(vision_angle::AbstractArray, pos::AbstractArray)
    vis_size = size(vision_angle)[2]
    pos_size = size(pos)[end]

    a = repeat(pos, outer = (1, vis_size))
    b = a .+ repeat(vision_angle, inner = (1, pos_size)) .* 4
    
    l = size(a)[end]
    c = pos_size
    
    d = line_sdf(
        repeat(a, outer = (1, c)),
        repeat(b, outer = (1, c)),
        repeat(pos, inner = (1, l))
    )
    return sum(reshape(smoothstep(1 .- d), pos_size, vis_size, :) .* (1 .- reshape(Diagonal(ones(pos_size)), pos_size, 1, pos_size)), dims = 3)
end

function render_base_env(b::BaseEnv)
    pos = b.agent.recur.state[1] |> cpu
    vis_angle = b.vision_angle |> cpu

    x = pos[1, :]
    y = pos[2, :]
    c = Raylib.RayColor(0/255, 255/255, 255/255, 255/255)
    function draw_circ(x, y)
        Raylib.DrawCircle(x, y, 16.0, c)
    end
    draw_circ.(Int.(round.(x * 16)), Int.(round.(y * 16)))
end