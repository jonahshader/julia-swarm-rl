using Raylib

include("math.jl")

const s_w = 1920
const s_h = 1080
const CIRCLES = 25
const LINES = 100
Raylib.InitWindow(s_w, s_h, "Tree Growing Simulator")
Raylib.SetTargetFPS(165)

# cool_tree = make_tree(7, 150.0, 0.75, 0.0, pi/8, 0.75)

cam = Raylib.RayCamera2D(0f0, 0f0, 0f0, 0f0, 0f0, 1f0)

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

lines = Vector{TestLine}()
circles = Vector{TestCircle}()

for i in 1:LINES
    push!(lines, TestLine(rand(Float32) * s_w, rand(Float32) * s_h, rand(Float32) * s_w, rand(Float32) * s_h))
end

for i in 1:CIRCLES
    push!(circles, TestCircle(rand(Float32) * s_w, rand(Float32) * s_h, rand(Float32) * 32))
end

function is_line_colliding(line)
    any(map(x->circle_line_intersecting(line.x1, line.y1, line.x2, line.y2, x.x, x.y, x.radius), circles))
end

while !Raylib.WindowShouldClose()
    if Raylib.IsKeyPressed(Raylib.KEY_W)
        cam.offset_y -= 1f0
    end
    if Raylib.IsKeyPressed(Raylib.KEY_S)
        cam.offset_y += 1f0
    end
    if Raylib.IsKeyPressed(Raylib.KEY_A)
        cam.offset_x -= 1f0
    end
    if Raylib.IsKeyPressed(Raylib.KEY_D)
        cam.offset_x += 1f0
    end

    circles[1].x = Raylib.GetMouseX()
    circles[1].y = Raylib.GetMouseY()

    # move circles
    for i in 2:length(circles)
        circles[i].x += rand(Float32) * 2f0 - 1f0
        circles[i].y += rand(Float32) * 2f0 - 1f0
    end

    Raylib.BeginDrawing()
        Raylib.BeginMode2D(cam)
            Raylib.ClearBackground(Raylib.RayColor(0/255,0/255,0/255,255/255))
            for i in 1:length(lines)
                if is_line_colliding(lines[i])
                    Raylib.DrawLine(Int(round(lines[i].x1)), Int(round(lines[i].y1)), Int(round(lines[i].x2)), Int(round(lines[i].y2)), Raylib.RayColor(255/255,0/255,0/255,255/255))
                else
                    Raylib.DrawLine(Int(round(lines[i].x1)), Int(round(lines[i].y1)), Int(round(lines[i].x2)), Int(round(lines[i].y2)), Raylib.RayColor(0/255,255/255,0/255,255/255))
                end
            end

            # render circles
            for i in 1:length(circles)
                Raylib.DrawCircle(Int(round(circles[i].x)), Int(round(circles[i].y)), Int(round(circles[i].radius)), Raylib.RayColor(255/255,255/255,255/255,255/255))
            end

        Raylib.EndMode2D()
    Raylib.EndDrawing()
end

Raylib.CloseWindow()