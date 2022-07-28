using LazyArrays

function make_matrices(num)
    mats = Vector{Matrix{Float32}}()
    for i in 1:num
        push!(mats, randn(Float32, 64, 64))
    end
    mats
end

function test(matrices)

    first = similar(matrices[1])
    for i in 2:length(matrices)
        
        first *= matrices[i]
    end
    first
end
