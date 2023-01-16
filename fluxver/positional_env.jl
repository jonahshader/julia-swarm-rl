using Flux
using CUDA

include("minimal_agent.jl")

struct PositionalEnv
    agent::MinimalAgent
end

Flux.@functor PositionalEnv
Flux.trainable(e::PositionalEnv) = (e.agent,)

function PositionalEnv(;batch_size::Integer = 64)
    a = MinimalAgent(2, batch_size) # pos, vel, targetpos
    return PositionalEnv(a)
end

# targetpos must match the size (2, batch_size)
function (m::PositionalEnv)(targetpos::AbstractMatrix)
    # input = vcat(m.agent.recur.state[2], targetpos .- m.agent.recur.state[1])
    input = m.agent.recur.state[1]
    m.agent(input)
end

function render(b::PositionalEnv)
    pos = b.agent.recur.state[1] |> cpu

    x = pos[1, :]
    y = pos[2, :]
    c = Raylib.RayColor(0/255, 255/255, 255/255, 255/255)
    function draw_circ(x, y)
        Raylib.DrawCircle(x, y, 16.0, c)
    end
    draw_circ.(Int.(round.(x * 16)), Int.(round.(y * 16)))
end

function reset!(m::PositionalEnv)
    reset!(m.agent)
end
