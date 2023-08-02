include("../base_env.jl")
include("../losses.jl")
include("../es.jl")

function minimise_sd_task(;b = BaseEnv())

    b_original = deepcopy(b)
    original_p = Flux.params(b)
    function param_to_fitness(p)
        for (s1,s2) in zip(b.agent.recur.state, b_original.agent.recur.state)
            s1 .= s2
        end
        for (p1, p2) in zip(original_p, p)
            p1 .= p2
        end

        for _ in 1:10
            b()
        end
    
        return sd(b)
    end

    es!(param_to_fitness, original_p, 10, noise_sd = 0.01f0)
    b
end
