include("base_env.jl")
include("losses.jl")

using Flux

function test_loss(b)
    b()
    sd(b)
end

function test_grad()
    b = BaseEnv()
    ps = Flux.params(b)
    opt = Adam()

    grad = gradient(() -> test_loss(b), ps)

    Flux.update!(opt, ps, grad)
    b
end