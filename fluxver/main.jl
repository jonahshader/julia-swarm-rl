include("base_env.jl")
using Raylib
using Flux

const s_w = 1920
const s_h = 1080

function run(; device = gpu, n=128, memory_size=128, vision_size=32)
    Raylib.InitWindow(s_w, s_h, "Swarm Reinforcement Learning Demo")
    Raylib.SetTargetFPS(165)

    # cam = Raylib.RayCamera2D(0f0, 0f0, 0f0, 0f0, 0f0, 1f0)
    cam = Raylib.RayCamera2D(s_w/2f0, s_h/2f0, 0f0, 0f0, 0f0, 0.5f0)

    env = BaseEnv(vision_size, memory_size, batch_size = n) |> device

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
                env() # update
                render_base_env(env)
            Raylib.EndMode2D()
            Raylib.DrawText("$(1/Raylib.GetFrameTime())", 32, 32, 16, Raylib.RayColor(1.0, 1.0, 1.0, 1.0))
        Raylib.EndDrawing()
    end
    
    
    
    Raylib.CloseWindow()
end