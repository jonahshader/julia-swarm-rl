include("base_env.jl")
include("losses.jl")
using Raylib
using Flux

const s_w = 1920
const s_h = 1080

function run(env::BaseEnv)
    Raylib.InitWindow(s_w, s_h, "Swarm Reinforcement Learning Demo")
    Raylib.SetTargetFPS(165)

    # cam = Raylib.RayCamera2D(0f0, 0f0, 0f0, 0f0, 0f0, 1f0)
    cam = Raylib.RayCamera2D(s_w/2f0, s_h/2f0, 0f0, 0f0, 0f0, 0.5f0)

    s = 4f0
    try
        while !Raylib.WindowShouldClose()
            if Raylib.IsKeyDown(Raylib.KEY_W)
                cam.offset_y += s
            end
            if Raylib.IsKeyDown(Raylib.KEY_S)
                cam.offset_y -= s
            end
            if Raylib.IsKeyDown(Raylib.KEY_A)
                cam.offset_x += s
            end
            if Raylib.IsKeyDown(Raylib.KEY_D)
                cam.offset_x -= s
            end
            # if Raylib.IsKeyPressed(Raylib.KEY_SPACE)
            #     env = BaseEnv(vision_size, memory_size, batch_size = n) |> device
            # end
            if Raylib.IsKeyPressed(Raylib.KEY_E)
                cam.zoom *= 1.5f0
            end
            if Raylib.IsKeyPressed(Raylib.KEY_Q)
                cam.zoom /= 1.5f0
            end
            Raylib.BeginDrawing()
                Raylib.BeginMode2D(cam)
                    Raylib.ClearBackground(Raylib.RayColor(0/255,0/255,0/255,255/255))
                    env() # update
                    render_base_env(env)
                Raylib.EndMode2D()
                Raylib.DrawText("$(1/Raylib.GetFrameTime())", 32, 32, 16, Raylib.RayColor(1.0, 1.0, 1.0, 1.0))
                Raylib.DrawText("sd: $(sd(env))", 32, 64, 16, Raylib.RayColor(1.0, 1.0, 1.0, 1.0))
            Raylib.EndDrawing()
        end
        
        Raylib.CloseWindow()

    catch e
        Raylib.CloseWindow()
        raise(e)
    end

end

function run(; device = gpu, n=128, memory_size=128, vision_size=32)
    env = BaseEnv(vision_size = vision_size, mem_size = memory_size, batch_size = n) |> device
    run(env)
end