using Revise
includet("agent.jl")
function test()
    a = Agent(32, 64, 1) |> gpu
    v = randn(32, 1, 1) |> gpu
    i = randn(1, 1) |> gpu
    
    a((v, i))
end


