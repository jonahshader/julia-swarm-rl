using Flux

struct AgentCore
    vision_chain::Chain
    main_chain::Chain
end

function AgentCore(vision_size::Integer, vision_features::Integer, mem_size::Integer, other_input_size::Integer)
    vision_chain = Chain(
        Flux.MeanPool((4,)),
        Flux.flatten,
        Flux.Dense(vision_features*vision_size÷4 => 128, Flux.swish) 
    )

    main_chain = Chain(
        Dense(128 + mem_size + other_input_size => 256, Flux.swish),
        Dense(256 => mem_size * 2 + 2 + 1, tanh)
    )

    return AgentCore(vision_chain, main_chain)
end

Flux.@functor AgentCore

struct Agent
    recur::Flux.Recur
end

function Agent(vision_size::Integer, vision_features::Integer, mem_size::Integer, other_input_size::Integer; batch_size::Integer = 1)
    core = AgentCore(vision_size, vision_features, mem_size, other_input_size)
    pos_init = randn(Float32, 2, batch_size)
    vel_init = zeros(Float32, 2, batch_size)
    mem_init = zeros(Float32, mem_size, batch_size)
    recur = Flux.Recur(core, (pos_init, vel_init, mem_init))
    return Agent(recur)
end

function reset!(agent::Agent; batch_size::Integer = agent.recur.state[1][end])
    pos_init = randn(Float32, 2, batch_size)
    vel_init = zeros(Float32, 2, batch_size)
    mem_init = zeros(Float32, mem_size, batch_size)
    agent.recur.state = (pos_init, vel_init, mem_init)
    nothing
end

function (m::Agent)(x)
    return m.recur(x)
end

Flux.@functor Agent

function (m::AgentCore)(h, x)
    pos, vel, memory = h
    vision, other = x

    model_output = m.main_chain(
        vcat(
            m.vision_chain(vision), 
            other, 
            memory))

    mem_size = size(memory)[1]
    
    mem_val = model_output[1:mem_size, :]
    mem_prop = model_output[mem_size+1:2*mem_size, :]
    mem_new = (mem_val .* (mem_prop .* .5f0 .+ .5f0)) .+ memory .* (mem_prop .* -.5f0 .+ .5f0)

    accel_dir = model_output[2*mem_size+1:2*mem_size+2, :]
    accel_mag = model_output[2*mem_size+2+1:2*mem_size+2+1, :]

    vel_new = (vel .+ accel_dir ./ sqrt.(sum(accel_mag .^ 2, dims=1))) .* 0.9f0
    pos_new = pos .+ vel_new .* (1f0/60)

    h_new = (pos_new, vel_new, mem_new)
    return h_new, h_new
end