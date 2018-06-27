#studies
@with_kw struct MetricStudy
    actions         = UniformActions()
    G               = TwoBumps()
    n_seeds::Int    = 500
    n_iters::Int    = 100
end
struct MetricStudyResult
    cum_regrets
    simple_regrets 
end

function generate_sim_q(sim::MetricStudy, alg)
    q = []
    for i in 1:sim.n_seeds
        b = alg(; actions=sim.actions, seed=i, n_iters=sim.n_iters, outs=Set([:simple_regret, :cum_regret])) 
        push!(q, BanditSim(b, sim.G))
    end
    q
end
function run_study(sim::MetricStudy)
    cum_regrets = [] 
    simple_regrets = [] 
    for alg in [RandomBandit, PWUCB, SBUCB]
		q = generate_sim_q(sim, alg)
		results = pmap(POMDPs.simulate, q)
		push!(cum_regrets, string(alg)=>mean(hcat([r.cum_regret for r in results]...),2))
		push!(simple_regrets, string(alg)=>mean(hcat([r.simple_regret for r in results]...),2))
    end
    MetricStudyResult(cum_regrets, simple_regrets)
end
Plots.plot(result::MetricStudyResult, metric::Symbol) = plot(result, Val{metric})
function Plots.plot(result::MetricStudyResult, ::Type{Val{:cum_regret}})
    p = plot(title="Cumulative regret")
    for (alg,cum_regret) in result.cum_regrets
        plot!(p, cum_regret, label=alg)
    end
    p
end
function Plots.plot(result::MetricStudyResult, ::Type{Val{:simple_regret}})
    p = plot(title="Simple regret")
    for (alg,simple_regret) in result.simple_regrets
        plot!(p, simple_regret, label=alg)
    end
    p
end
