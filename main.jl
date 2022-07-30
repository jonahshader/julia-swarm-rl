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
include("raylib/raylib.jl")

const s_w = 1920
const s_h = 1080
InitWindow(s_w, s_h, "Tree Growing Simulator")
SetTargetFPS(165)

# cool_tree = make_tree(7, 150.0, 0.75, 0.0, pi/8, 0.75)

const creatures = make_creatures(8192, Float32(s_w), Float32(s_h), [100, 100], 512, 8, device=:gpu, rand_memory=true)
while !WindowShouldClose()
    BeginDrawing()
        ClearBackground(Color(0, 0, 0, 255))
        update_creatures!(creatures)
        update_creatures!(creatures)
        update_creatures!(creatures)
        render_creatures(creatures)

    EndDrawing()
end

CloseWindow()