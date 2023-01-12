using Flux
using Flux: Optimise
using Zygote

function es!(params_to_fitness, base_params, generations; population_size = 32, noise_sd = 0.01f0, opt = Adam())
    positive_size = population_size ÷ 2
    for i in 1:generations
        println("Generation $i")
        es_generation!(params_to_fitness, base_params, positive_size, noise_sd, opt)
    end
    nothing
end

function similar_randn(p)
    s = Vector{AbstractArray{Float32}}()
    for v in p
        push!(s, randn(size(v)))
    end
    s
end

function es_generation!(params_to_fitness, base_params, positive_size, noise_sd, opt)
    noise = [similar_randn(base_params) .* noise_sd for _ in 1:positive_size]
    population = vcat([base_params .+ n for n in noise], [base_params .- n for n in noise])
    pop_size = length(population)
    pop_fitness = [params_to_fitness(p) for p in population]
    sorted_indices = sortperm(pop_fitness)
    grad_approx = sum(population[sorted_indices] .* ((collect(0:pop_size-1) ./ (pop_size-1f0)) .- .5f0)) ./ pop_size
    buff = Zygote.Buffer{Any, Vector{Any}}(grad_approx |> deepcopy, false)
    grad_approx = Flux.Params(buff, base_params.params |> deepcopy)
    # Flux.update!(opt, base_params, (grad_approx, ))
    # Adam
    # base_params .+= apply!(opt, base_params, grad_approx)
    # base_params .-= grad_approx

    delta = grad_approx .* 0.1f0
    for (p1, p2) in zip(base_params, delta)
        p1 .+= p2
    end
    println(sum(pop_fitness[sorted_indices]))
    
    # opt(base_params, grad_approx)
    nothing
end