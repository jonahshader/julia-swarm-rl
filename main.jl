include("systems/creature.jl")
using Raylib

const s_w = 1920
const s_h = 1080

function run(;device=:gpu, n=8192, memory_size=512)
    Raylib.InitWindow(s_w, s_h, "Tree Growing Simulator")
    Raylib.SetTargetFPS(165)
    
    # cool_tree = make_tree(7, 150.0, 0.75, 0.0, pi/8, 0.75)
    
    cam = Raylib.RayCamera2D(0f0, 0f0, 0f0, 0f0, 0f0, 1f0)
    
    creatures = make_creatures(n, Float32(s_w), Float32(s_h), [100, 100], memory_size, 8, device=device, rand_memory=true)
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
        Raylib.BeginDrawing()
            Raylib.BeginMode2D(cam)
                Raylib.ClearBackground(Raylib.RayColor(0/255,0/255,0/255,255/255))
                update_creatures!(creatures)
                update_creatures!(creatures)
                update_creatures!(creatures)
                render_creatures(creatures)
            Raylib.EndMode2D()
        Raylib.EndDrawing()
    end
    
    Raylib.CloseWindow()
end
