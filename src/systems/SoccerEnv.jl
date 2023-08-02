include("agent.jl")
mutable struct SoccerEnv
    team1::Agent
    team2::Agent
    ball::AbstractArray

end