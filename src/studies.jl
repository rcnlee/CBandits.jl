#studies
#############
@with_kw struct MetricStudy
    actiondistr     = UniformActions()
    G               = TwoBumps()
    n_seeds::Int    = 500
    n_iters::Int    = 100
end
struct MetricStudyResult
    cum_regrets
    simple_regrets 
end

function generate_sim_q(study::MetricStudy, alg)
    q = []
    for i in 1:study.n_seeds
        b = alg(; actiondistr=study.actiondistr, seed=i, n_iters=study.n_iters, 
                outs=Set([:simple_regret, :cum_regret])) 
        push!(q, BanditSim(b, study.G))
    end
    q
end
function run_study(study::MetricStudy)
    cum_regrets = [] 
    simple_regrets = [] 
    for alg in [RandomBandit, PWUCB, SBUCB, GPUCB]
		q = generate_sim_q(study, alg)
		results = pmap(POMDPs.simulate, q)
		push!(cum_regrets, string(alg)=>mean(hcat([r.cum_regret for r in results]...),2))
		push!(simple_regrets, string(alg)=>mean(hcat([r.simple_regret for r in results]...),2))
    end
    MetricStudyResult(cum_regrets, simple_regrets)
end
Plots.plot(result::MetricStudyResult, metric::Symbol) = plot(result, Val{metric})
function Plots.plot(result::MetricStudyResult, ::Type{Val{:cum_regret}})
    p = plot(; ylabel="cumulative regret", xlabel="number of samples", legend=:topleft)
    for (alg,cum_regret) in result.cum_regrets
        plot!(p, cum_regret, label=string(alg))
    end
    p
end
function Plots.plot(result::MetricStudyResult, ::Type{Val{:simple_regret}})
    p = plot(; ylabel="simple regret", xlabel="number of samples")
    for (alg,simple_regret) in result.simple_regrets
        plot!(p, simple_regret, label=string(alg))
    end
    p
end

#################
@with_kw struct SweepStudy
    actiondistr     = UniformActions()
    G               = NarrowBump()
    n_seeds::Int     = 100
    n_iters::Int     = 100
    ks              = linspace(0.0, 3.0, 10)
    αs              = linspace(0.0, 2.0, 10) 
end

function generate_sim_q(study::SweepStudy, alg)
    q = []
    for i = 1:study.n_seeds
        b = RandomBandit(; actiondistr=study.actiondistr, seed=i, n_iters=study.n_iters, 
                         outs=Set([:simple_regret,:cum_regret]))
        push!(q, BanditSim(b, study.G))
    end
    for i = 1:study.n_seeds
        b = PWUCB(; actiondistr=study.actiondistr, seed=i, n_iters=study.n_iters, 
                         outs=Set([:simple_regret,:cum_regret]))
        push!(q, BanditSim(b, study.G))
    end
    for i = 1:study.n_seeds, k in study.ks, α in study.αs
        b = alg(; actiondistr=study.actiondistr, seed=i, n_iters=study.n_iters, k=k, α=α,
                         outs=Set([:simple_regret,:cum_regret]))
        push!(q, BanditSim(b, study.G))
    end
    q
end
function run_study(study::SweepStudy, alg)
    q = generate_sim_q(study, alg)
    data = run_parallel(q) do sim, result
        return vcat(metadata(sim.b), 
                    [:mean_cum_regret=>mean(result.cum_regret), 
                     :mean_simple_regret=>mean(result.simple_regret)])
    end
    data
end

#############
@with_kw struct GPkMetricStudy
    actiondistr     = UniformActions()
    G               = TwoBumps()
    n_seeds::Int    = 500
    n_iters::Int    = 100
    ks::Vector{Int} = [100, 50, 20, 10, 5, 1]
end
struct GPkMetricStudyResult
    cum_regrets
    simple_regrets 
end
function generate_sim_q(study::GPkMetricStudy, alg; kwargs...)
    q = []
    for i in 1:study.n_seeds
        b = alg(; actiondistr=study.actiondistr, seed=i, n_iters=study.n_iters, outs=Set([:simple_regret, :cum_regret]), 
                kwargs...) 
        push!(q, BanditSim(b, study.G))
    end
    q
end
function run_study(study::GPkMetricStudy)
    cum_regrets = [] 
    simple_regrets = [] 
    for alg in [RandomBandit, PWUCB, SBUCB, GPUCBGrid]
		q = generate_sim_q(study, alg)
		results = pmap(POMDPs.simulate, q)
		push!(cum_regrets, string(alg)=>mean(hcat([r.cum_regret for r in results]...),2))
		push!(simple_regrets, string(alg)=>mean(hcat([r.simple_regret for r in results]...),2))
    end
    for k in study.ks 
		q = generate_sim_q(study, GPUCB; k=k)
		results = pmap(POMDPs.simulate, q)
        push!(cum_regrets, "$(string(GPUCB))-$k"=>mean(hcat([r.cum_regret for r in results]...),2))
        push!(simple_regrets, "$(string(GPUCB))-$k"=>mean(hcat([r.simple_regret for r in results]...),2))
    end
    GPkMetricStudyResult(cum_regrets, simple_regrets)
end
Plots.plot(result::GPkMetricStudyResult, metric::Symbol) = plot(result, Val{metric})
function Plots.plot(result::GPkMetricStudyResult, ::Type{Val{:cum_regret}})
    p = plot(; xlabel="number of samples", ylabel="cumulative regret", legend=:topleft)
    for (alg,cum_regret) in result.cum_regrets
        plot!(p, cum_regret, label=string(alg))
    end
    p
end
function Plots.plot(result::GPkMetricStudyResult, ::Type{Val{:simple_regret}})
    p = plot(; xlabel="number of samples", ylabel="simple regret")
    for (alg,simple_regret) in result.simple_regrets
        plot!(p, simple_regret, label=string(alg))
    end
    p
end

################
@with_kw struct GPOptimMetricStudy
    actiondistr     = UniformActions()
    G               = TwoBumps()
    n_seeds::Int    = 500
    n_iters::Int    = 100
    mean_inits::Vector{Int} = [8.0,5.0,3.0,2.0,0.0]
end
struct GPOptimMetricStudyResult
    cum_regrets
    simple_regrets 
end
function generate_sim_q(study::GPOptimMetricStudy, alg; kwargs...)
    q = []
    for i in 1:study.n_seeds
        b = alg(; actiondistr=study.actiondistr, seed=i, n_iters=study.n_iters, 
                outs=Set([:simple_regret, :cum_regret]), kwargs...) 
        push!(q, BanditSim(b, study.G))
    end
    q
end
function run_study(study::GPOptimMetricStudy)
    cum_regrets = [] 
    simple_regrets = [] 
    for alg in [RandomBandit, PWUCB, SBUCB, GPUCBGrid]
		q = generate_sim_q(study, alg)
		results = pmap(POMDPs.simulate, q)
		push!(cum_regrets, string(alg)=>mean(hcat([r.cum_regret for r in results]...),2))
		push!(simple_regrets, string(alg)=>mean(hcat([r.simple_regret for r in results]...),2))
    end
    for minit in study.mean_inits 
		q = generate_sim_q(study, GPUCB; mean_init=minit)
		results = pmap(POMDPs.simulate, q)
        push!(cum_regrets, "$(string(GPUCB))-$minit"=>mean(hcat([r.cum_regret for r in results]...),2))
        push!(simple_regrets, "$(string(GPUCB))-$minit"=>mean(hcat([r.simple_regret for r in results]...),2))
    end
    GPOptimMetricStudyResult(cum_regrets, simple_regrets)
end
Plots.plot(result::GPOptimMetricStudyResult, metric::Symbol) = plot(result, Val{metric})
function Plots.plot(result::GPOptimMetricStudyResult, ::Type{Val{:cum_regret}})
    p = plot(; xlabel="number of samples", ylabel="cumulative regret", legend=:topleft)
    for (alg,cum_regret) in result.cum_regrets
        plot!(p, cum_regret, label=string(alg))
    end
    p
end
function Plots.plot(result::GPOptimMetricStudyResult, ::Type{Val{:simple_regret}})
    p = plot(; xlabel="number of samples", ylabel="simple regret")
    for (alg,simple_regret) in result.simple_regrets
        plot!(p, simple_regret, label=string(alg))
    end
    p
end

####
@with_kw struct MetricStudy2D
    actiondistr     = UniformActions2D()
    G               = WideBump2D()
    n_seeds::Int    = 20 
    n_iters::Int    = 200
end
struct MetricStudy2DResult
    cum_regrets
    simple_regrets 
end

function generate_sim_q(study::MetricStudy2D, alg; kwargs...)
    q = []
    for i in 1:study.n_seeds
        b = alg(; actiondistr=study.actiondistr, seed=i, n_iters=study.n_iters, 
                outs=Set([:simple_regret, :cum_regret]), kwargs...)
        push!(q, BanditSim(b, study.G))
    end
    q
end
function run_study(study::MetricStudy2D)
    cum_regrets = [] 
    simple_regrets = [] 
    for alg in [RandomBandit, PWUCB, SBUCB]
		q = generate_sim_q(study, alg)
		results = pmap(POMDPs.simulate, q)
		push!(cum_regrets, string(alg)=>mean(hcat([r.cum_regret for r in results]...),2))
		push!(simple_regrets, string(alg)=>mean(hcat([r.simple_regret for r in results]...),2))
    end
    for alg in [GPUCBGrid, GPUCB]
        q = generate_sim_q(study, alg, log_length_scale=-0.5, grid=GridActions2D())
		results = pmap(POMDPs.simulate, q)
		push!(cum_regrets, string(alg)=>mean(hcat([r.cum_regret for r in results]...),2))
		push!(simple_regrets, string(alg)=>mean(hcat([r.simple_regret for r in results]...),2))
    end
    for alg in [GPUCBGrid, GPUCB]
        q = generate_sim_q(study, alg, mean_init=study.G.g_max+1.0, log_length_scale=-0.5, grid=GridActions2D())
		results = pmap(POMDPs.simulate, q)
		push!(cum_regrets, "$(string(alg))-optim"=>mean(hcat([r.cum_regret for r in results]...),2))
		push!(simple_regrets, "$(string(alg))-optim"=>mean(hcat([r.simple_regret for r in results]...),2))
    end
    MetricStudy2DResult(cum_regrets, simple_regrets)
end
Plots.plot(result::MetricStudy2DResult, metric::Symbol) = plot(result, Val{metric})
function Plots.plot(result::MetricStudy2DResult, ::Type{Val{:cum_regret}})
    p = plot(; ylabel="cumulative regret", xlabel="number of samples", legend=:topleft)
    for (alg,cum_regret) in result.cum_regrets
        plot!(p, cum_regret, label=string(alg))
    end
    p
end
function Plots.plot(result::MetricStudy2DResult, ::Type{Val{:simple_regret}})
    p = plot(; ylabel="simple regret", xlabel="number of samples")
    for (alg,simple_regret) in result.simple_regrets
        plot!(p, simple_regret, label=string(alg))
    end
    p
end

######
@with_kw struct LengthScaleStudy2D
    actiondistr     = UniformActions2D()
    G               = WideBump2D()
    n_seeds::Int    = 20 
    n_iters::Int    = 200
    ls::Vector{Float64} = [0.05, 0.1, 0.3, 0.6, 0.8, 1.0, 1.5]
end
struct LengthScaleStudy2DResult
    cum_regrets
    simple_regrets 
end

function generate_sim_q(study::LengthScaleStudy2D, alg; kwargs...)
    q = []
    for i in 1:study.n_seeds
        b = alg(; actiondistr=study.actiondistr, seed=i, n_iters=study.n_iters, 
                outs=Set([:simple_regret, :cum_regret]), kwargs...)
        push!(q, BanditSim(b, study.G))
    end
    q
end
function run_study(study::LengthScaleStudy2D)
    cum_regrets = [] 
    simple_regrets = [] 
    for alg in [RandomBandit, PWUCB, SBUCB]
		q = generate_sim_q(study, alg)
		results = pmap(POMDPs.simulate, q)
		push!(cum_regrets, string(alg)=>mean(hcat([r.cum_regret for r in results]...),2))
		push!(simple_regrets, string(alg)=>mean(hcat([r.simple_regret for r in results]...),2))
    end
    for alg in [GPUCBGrid]
        q = generate_sim_q(study, alg, log_length_scale=-0.5, grid=GridActions2D())
		results = pmap(POMDPs.simulate, q)
		push!(cum_regrets, string(alg)=>mean(hcat([r.cum_regret for r in results]...),2))
		push!(simple_regrets, string(alg)=>mean(hcat([r.simple_regret for r in results]...),2))
    end
    for len in study.ls, alg in [GPUCB]
        q = generate_sim_q(study, alg, log_length_scale=log(len), grid=GridActions2D())
		results = pmap(POMDPs.simulate, q)
		push!(cum_regrets, "$(string(alg))-l=$len"=>mean(hcat([r.cum_regret for r in results]...),2))
		push!(simple_regrets, "$(string(alg))-l=$len"=>mean(hcat([r.simple_regret for r in results]...),2))
    end
    LengthScaleStudy2DResult(cum_regrets, simple_regrets)
end
Plots.plot(result::LengthScaleStudy2DResult, metric::Symbol) = plot(result, Val{metric})
function Plots.plot(result::LengthScaleStudy2DResult, ::Type{Val{:cum_regret}})
    p = plot(; ylabel="cumulative regret", xlabel="number of samples", legend=:topleft)
    for (alg,cum_regret) in result.cum_regrets
        plot!(p, cum_regret, label=string(alg))
    end
    p
end
function Plots.plot(result::LengthScaleStudy2DResult, ::Type{Val{:simple_regret}})
    p = plot(; ylabel="simple regret", xlabel="number of samples")
    for (alg,simple_regret) in result.simple_regrets
        plot!(p, simple_regret, label=string(alg))
    end
    p
end
