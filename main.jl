# include("raylib/raylib.jl")


# const s_w = 1920
# const s_h = 1080
# InitWindow(s_w, s_h, "Tree Growing Simulator")
# SetTargetFPS(165)

# # cool_tree = make_tree(7, 150.0, 0.75, 0.0, pi/8, 0.75)
# while !WindowShouldClose()
#     BeginDrawing()
#         ClearBackground(Color(0, 0, 0, 255))
        
#     EndDrawing()
# end


include("systems/creature.jl")
using Raylib

const s_w = 1920
const s_h = 1080
Raylib.InitWindow(s_w, s_h, "Tree Growing Simulator")
Raylib.SetTargetFPS(165)

# cool_tree = make_tree(7, 150.0, 0.75, 0.0, pi/8, 0.75)

cam = Raylib.RayCamera2D(0f0, 0f0, 0f0, 0f0, 0f0, 1f0)

const creatures = make_creatures(4096, Float32(s_w), Float32(s_h), [100, 100], 512, 8, device=:gpu, rand_memory=true)
while !Raylib.WindowShouldClose()
    if Raylib.IsKeyDown(Raylib.KEY_W)
        cam.offset_y += 3f0
    end
    if Raylib.IsKeyDown(Raylib.KEY_S)
        cam.offset_y -= 3f0
    end
    if Raylib.IsKeyDown(Raylib.KEY_A)
        cam.offset_x += 3f0
    end
    if Raylib.IsKeyDown(Raylib.KEY_D)
        cam.offset_x -= 3f0
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