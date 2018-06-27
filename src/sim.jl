struct BanditSim
    b::Bandit
    G::ObjectiveFunc 
    metadata::Dict
end
BanditSim(b::Bandit, G::ObjectiveFunc) = BanditSim(b, G, Dict(:algorithm=>string(typeof(b)), 
                                                              :objective=>string(typeof(G))))
POMDPs.simulate(sim::BanditSim) = POMDPs.solve(sim.b, sim.G)
