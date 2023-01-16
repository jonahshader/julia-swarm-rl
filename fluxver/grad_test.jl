include("positional_env.jl")
include("losses.jl")

using Flux

function test_loss(b, x)
    b(x)
    sd(b)
end

function test_grad()
    b = PositionalEnv()
    x = randn(Float32, 2, 64)
    ps = Flux.params(b)
    opt = Adam()

    grad = gradient(() -> test_loss(b, x), ps)

    Flux.update!(opt, ps, grad)
    b
end