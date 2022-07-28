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
