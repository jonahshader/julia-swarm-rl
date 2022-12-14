
using LinearAlgebra

function circle_line_intersecting(ax::Number, ay::Number, bx::Number, by::Number, cx::Number, cy::Number, r::Number)
    area2 = abs((bx-ax) * (cy-ay) - (cx-ax) * (by-ay))
    lab = sqrt((bx-ax)^2 + (by-ay)^2)
    h = area2/lab
    h < r
end

# TODO: make version that takes in a point and direction instead of two points.
function circle_line_intersecting(ax::AbstractArray, ay::AbstractArray, bx::AbstractArray, by::AbstractArray, cx::AbstractArray, cy::AbstractArray, r::AbstractArray)
    area2 = abs.((bx.-ax) .* (cy.-ay) .- (cx.-ax) .* (by.-ay))
    lab = sqrt.((bx.-ax).^2 .+ (by.-ay).^2)
    h = area2./lab
    h .< r
end


mat_dot(a, b) = sum(a.*b, dims = 1)

function line_sdf(a::AbstractArray, b::AbstractArray, p::AbstractArray)
    h = clamp.(mat_dot(p.-a, b.-a) ./ mat_dot(b.-a, b.-a), 0, 1)
    sqrt.(sum((p.-a.-(b.-a).*h) .^ 2, dims = 1))
end

function smoothstep(x)
    xc = clamp(x, 0, 1)
    xc * xc * (3 - 2 * xc)
end

function smoothstep(x::AbstractArray)
    xc = clamp.(x, 0, 1)
    xc .* xc .* (3 .- 2 .* xc)
end

function intersect_ray_circle(p::T, d::T, s::Tuple{T, AbstractFloat})::Tuple{Bool, AbstractFloat, T} where {T <: AbstractArray}
    (sc, sr) = s
    m = p - sc
    b = dot(m, d)
    c = dot(m, m) - sr*sr

    if (c > 0 && b > 0) 
        return (false, 0.0, zero(p))
    end
    
    discr = b*b - c
    if (discr < 0)
        return (false, 0.0, zero(p))
    end

    t = -b - sqrt(discr)

    if (t < 0)
        t = 0.0
    end
    q = p + t*d
    return (true, t, q)
end