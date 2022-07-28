include("creature.jl")
mutable struct SoccerEnv
    team1::Creature
    team2::Creature
    ball::AbstractArray

end