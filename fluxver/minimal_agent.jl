using Flux

struct MinimalAgentCore
    main_chain::Chain
end

function MinimalAgentCore()
    main_chain = Chain(
        Dense(4 => 128, Flux.swish),
        Dense(128 => 128, Flux.swish),
        Dense(128 => 3)
    )
    return MinimalAgentCore(main_chain)
end

Flux.@functor MinimalAgentCore

struct MinimalAgent
    recur::Flux.Recur
end

function make_init_state(batch_size)
    pos_init = randn(Float32, 2, batch_size) * 16f0
    vel_init = zeros(Float32, 2, batch_size)
    pos_init, vel_init
end

function MinimalAgent()