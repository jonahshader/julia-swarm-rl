using Flux

output_size = 5
input_size = 2
Wxh = randn(Float32, output_size, input_size)
Whh = randn(Float32, output_size, output_size)
b   = randn(Float32, output_size)

function rnn_cell(h, x)
    h = tanh.(Wxh * x .+ Whh * h .+ b)
    return h, h
end

x = rand(Float32, input_size)
h = rand(Float32, output_size)

h, y = rnn_cell(h, x)

m = Flux.Recur(rnn_cell, h)