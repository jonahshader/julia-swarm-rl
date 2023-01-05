using Flux

function es(params_to_fitness, initial_params; population_size = 32, noise_sd = 0.01, opt = Adam())
    positive_size = population_size ÷ 2
    
end

function similar_randn(p)
    s = Vector{AbstractArray{Float32}}()
    for v in p
        push!(s, randn(size(v)))
    end
    s
end

function es_generation(params_to_fitness, base_params, positive_size, noise_sd, opt)
    noise = [similar_randn(base_params) .* noise_sd for _ in 1:positive_size]
    population = vcat([base_params .+ n for n in noise], [base_params .- n for n in noise])
    pop_fitness = [(p, params_to_fitness(p)) for p in population]
end