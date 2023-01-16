using Flux
using Random

struct MinimalAgentCore
    main_chain::Chain
end

Flux.@functor MinimalAgentCore

function MinimalAgentCore(other_input_size::Integer)
    main_chain = Chain(
        Dense(other_input_size => 128, Flux.swish),
        Dense(128 => 128, Flux.swish),
        Dense(128 => 3, tanh)
    )
    return MinimalAgentCore(main_chain)
end

struct MinimalAgent
    recur::Flux.Recur
end

function (m::MinimalAgent)(x)
    return m.recur(x)
end

Flux.@functor MinimalAgent

function make_init_state(batch_size)
    pos_init = randn(Float32, 2, batch_size) * 16f0
    vel_init = zeros(Float32, 2, batch_size)
    pos_init, vel_init
end

function reset!(m::MinimalAgent)
    # device = if m.recur.state[1] |> typeof <: CuArray
    #     gpu
    # else
    #     cpu
    # end
    # m.recur.state = make_init_state(size(m.recur.state[1])[end]) .|> device
    randn!(m.recur.state[1])
    m.recur.state[1] .*= 16f0
    m.recur.state[2] .= 0
end

function MinimalAgent(other_input_size::Integer, batch_size::Integer)
    core = MinimalAgentCore(other_input_size)
    recur = Flux.Recur(core, make_init_state(batch_size))
    return MinimalAgent(recur)
end



function (m::MinimalAgentCore)(h, x)
    pos, vel = h

    model_output = m.main_chain(x)

    accel_dir = model_output[1:2, :]
    accel_mag = model_output[3:3, :] .* 5

    vel_new = (vel .+ accel_dir .* accel_mag ./ sqrt.(sum(accel_dir .^ 2, dims=1))) .* 0.9f0
    pos_new = pos .+ vel_new .* (1f0/60)

    h_new = (pos_new, vel_new)
    return h_new, h_new
end