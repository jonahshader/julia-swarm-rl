using Flux

include("agent.jl")
include("../systems/math.jl")

struct BaseEnv
    agent::Agent
    vision_angle::AbstractArray
    cancel_diag::AbstractArray
end

function BaseEnv(vision_size::Integer, mem_size::Integer; batch_size::Integer = 1)
    vision_features = 1
    other_input_size = 4

    a = Agent(vision_size, vision_features, mem_size, other_input_size, batch_size=batch_size)
    vision_angle = vcat([[cos(i*2pi/vision_size) sin(i*2pi/vision_size)] for i in 0:vision_size-1]...)'
    cancel_diag = a = [j==k ? 0 : 1 for i=1:vision_size, j=1:batch_size, k=1:batch_size]
    BaseEnv(a, vision_angle, cancel_diag)
end

function (m::BaseEnv)()
    # run iteration
    
    # run the agent with inputs, return its output
    m.agent()
end

function update_vision_soft(vision_angle::AbstractArray, cancel_diag::AbstractArray, pos::AbstractArray)
    vis_size = size(vision_angle)[2]
    println(vis_size)
    pos_size = size(pos)[end]
    println(pos_size)
    println(size(vision_angle))
    println(size(pos))

    a = repeat(pos, inner = (1, vis_size))
    b = a .+ repeat(vision_angle, 1, pos_size) .* 1024
    
    l = size(a)[end]
    c = pos_size
    
    d = line_sdf(
        repeat(a, 1, c),
        repeat(b, 1, c),
        repeat(pos, inner = (1, l))
    )
    return reshape(smoothstep(1 .- d), vis_size, pos_size, :) .* cancel_diag
end

function update_vision_soft(vision_angle::AbstractArray, cancel_diag::AbstractArray, pos::AbstractArray, circles::AbstractArray)
    vis_size = size(vision_angle)[2]
    println(vis_size)
    pos_size = size(pos)[end]
    println(pos_size)
    println(size(vision_angle))
    println(size(pos))

    a = repeat(pos, inner = (1, vis_size))
    b = a .+ repeat(vision_angle, 1, pos_size) .* 1024
    
    l = size(a)[end]
    c = size(cirlces)[end]
    
    d = line_sdf(
        repeat(a, 1, c),
        repeat(b, 1, c),
        repeat(circles, inner = (1, l))
    )
    return reshape(smoothstep(1 .- d), vis_size, pos_size, :) .* cancel_diag
end

Flux.@functor BaseEnv
Flux.trainable(m::BaseEnv) = (m.agent,)