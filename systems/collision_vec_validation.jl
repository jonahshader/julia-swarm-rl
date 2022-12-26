include("math.jl")

struct TestLine
    x1::AbstractFloat
    y1::AbstractFloat
    x2::AbstractFloat
    y2::AbstractFloat
end

mutable struct TestCircle
    x::AbstractFloat
    y::AbstractFloat
    radius::AbstractFloat
end

lines_vec = randn(Float32, 32, 4)
circles_vec = hcat(randn(Float32, 1024, 2), fill(0.2f0, 1024, 1))

lines = Vector{TestLine}()
circles = Vector{TestCircle}()

for i in 1:size(lines_vec)[1]
    push!(lines, TestLine(lines_vec[i, 1], lines_vec[i, 2], lines_vec[i, 3], lines_vec[i, 4]))
end

for i in 1:size(circles_vec)[1]
    push!(circles, TestCircle(circles_vec[i, 1], circles_vec[i, 2], circles_vec[i, 3]))
end

function is_line_colliding(line::TestLine)
    any(map(x->circle_line_intersecting(line.x1, line.y1, line.x2, line.y2, x.x, x.y, x.radius), circles))
end

line_circle_collisions = zeros(Bool, length(lines), length(circles))
for i in 1:length(lines)
    for j in 1:length(circles)
        line = lines[i]
        circle = circles[j]
        line_circle_collisions[i, j] = circle_line_intersecting(line.x1, line.y1, line.x2, line.y2, circle.x, circle.y, circle.radius)
    end
end

# line_circle_collisions_vec = zeros(Bool, length(lines), length(circles))
l = length(lines)
c = length(circles)
line_circle_collisions_vec = circle_line_intersecting(
    repeat(lines_vec[:, 1], 1, c), 
    repeat(lines_vec[:, 2], 1, c), 
    repeat(lines_vec[:, 3], 1, c), 
    repeat(lines_vec[:, 4], 1, c), 
    repeat(circles_vec[:, 1], 1, l)', 
    repeat(circles_vec[:, 2], 1, l)', 
    repeat(circles_vec[:, 3], 1, l)')

# ok it works

