using Flux

struct AgentCore
    vision_chain::Chain
    main_chain::Chain
end

function AgentCore(vision_size::Integer, vision_downscale::Integer, vision_features::Integer, mem_size::Integer, other_input_size::Integer)
    vision_chain = Chain(
        Flux.MeanPool((vision_downscale,)),
        Flux.flatten,
        Flux.Dense(vision_features*vision_size÷vision_downscale => 128, Flux.swish) 
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

function make_init_state(mem_size, batch_size)
    pos_init = randn(Float32, 2, batch_size) * 16f0
    vel_init = zeros(Float32, 2, batch_size)
    mem_init = zeros(Float32, mem_size, batch_size)
    pos_init, vel_init, mem_init
end

function Agent(vision_size::Integer, vision_features::Integer, mem_size::Integer, other_input_size::Integer, batch_size::Integer, vision_downscale::Integer)
    core = AgentCore(vision_size, vision_downscale, vision_features, mem_size, other_input_size)
    recur = Flux.Recur(core, make_init_state(mem_size, batch_size))
    return Agent(recur)
end

function reset!(agent::Agent; batch_size::Integer = size(agent.recur.state[1])[end], mem_size::Integer = size(agent.recur.state[3])[1])
    agent.recur.state = make_init_state(mem_size, batch_size)
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
    accel_mag = model_output[2*mem_size+2+1:2*mem_size+2+1, :] .* 5

    vel_new = (vel .+ accel_dir .* accel_mag ./ sqrt.(sum(accel_dir .^ 2, dims=1))) .* 0.9f0
    pos_new = pos .+ vel_new .* (1f0/60)

    h_new = (pos_new, vel_new, mem_new)
    return h_new, h_new
end