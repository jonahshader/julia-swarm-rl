
using LinearAlgebra

# function circle_line_intersecting(ax, ay, bx, by, cx, cy, r)
#     area2 = abs((bx-ax) * (cy-ay) - (cx-ax) * (by-ay))
#     lab = sqrt((bx-ax)^2 + (by-ay)^2)
#     h = area2/lab
#     h < r
# end

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