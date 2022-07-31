using Raylib

struct Circle
    x::Float32
    y::Float32
    radius::Float32
end

struct Ray2
    x::Float32
    y::Float32
    angle::Float32
end

function line_height(x::Float32, ray::Ray2)
    (x - ray.x) * tan(ray.angle) + ray.y
end

function intersect(ray::Ray2, circle::Circle)
    xc = ray.x - circle.x
    yc = ray.y - circle.y
    slope = tan(ray.angle)


end