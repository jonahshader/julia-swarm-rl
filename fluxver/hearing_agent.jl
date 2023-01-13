using Flux

struct HearingAgentCore
    main_chain::Chain
end

function AgentCore(mem_size::Integer, sound_size::Integer, position_encode_size::Integer, other_input_size::Integer)
    main_chain = Chain(
        Dense(3 * sound_size + mem_size + position_encode_size + other_input_size => 256, Flux.swish),
        Dense(256 => 256, Flux.swish),
        Dense(256 => )
    )
end