struct BanditSim
    b::Bandit
    G::ObjectiveFunc 
    metadata::Dict
end
function BanditSim(b::Bandit, G::ObjectiveFunc) 
    BanditSim(b, G, Dict(:sim_algorithm=>string(b), :sim_objective=>string(G)))
end
POMDPs.simulate(sim::BanditSim) = POMDPs.solve(sim.b, sim.G)
