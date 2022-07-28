
mutable struct Environment
    terrain::AbstractMatrix{Bool}
end

function Environment(width::Int, height::Int)
    terrain = zeros(Bool, width, height)
    Environment(terrain)
end

