include("../positional_env.jl")

using Zygote

function goto_pos_task(;b = PositionalEnv(), steps = 10, iterations = 100, opt = Adam())
    device = if b.agent.recur.state[1] |> typeof <: CuArray
        gpu
    else
        cpu
    end
    ps = Flux.params(b)

    function loss(b, x)
        l = 0f0
        for _ in 1:steps
            b(x)
            l += sum((b.agent.recur.state[1] .- x) .^ 2) / size(x)[end]
        end
        l /= steps
        println(l)
        return l
    end

    for i in 1:iterations
        # x = randn(Float32, 2, size(b.agent.recur.state[1])[end]) .* 64f0 |> device
        x = zeros(Float32, 2, size(b.agent.recur.state[1])[end]) |> device
        reset!(b)
        grad = gradient(() -> loss(b, x), ps)
        Flux.update!(opt, ps, grad)
    end
    reset!(b)
    return b
end