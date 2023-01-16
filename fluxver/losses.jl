function sd(b)
    mean = sum(b.agent.recur.state[1], dims = 2)
    sd = sum((b.agent.recur.state[1] .- mean) .^ 2, dims = 2) ./ length(mean)
    return sum(sd)
end