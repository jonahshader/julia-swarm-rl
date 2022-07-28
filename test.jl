include("systems/nn.jl")

using BenchmarkTools

nn = make_net([5, 10, 10, 3], tanh)

input = randn(Float32, 5)

@btime forward!(nn, input)

mat_input = randn(Float32, 5, 1024)

change_batch_size!(nn, 1024)

@btime forward!(nn, mat_input)

# test change