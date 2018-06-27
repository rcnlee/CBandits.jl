#Algorithms
abstract type Bandit end
abstract type BanditResult end
@with_kw mutable struct RandomBandit <: Bandit
    actions::ActionFunc     = UniformActions()
    seed::Int               = 0
    n_iters::Int            = 100
    outs::Set{Symbol}       = Set{Symbol}()
end
metadata(b::RandomBandit) = (:algorithm=>"Random", :seed=>b.seed, :n_iters=>b.n_iters,)
Base.string(::Type{RandomBandit}) = "Random"
struct RandomBanditResult <: BanditResult
    actions::Vector{Float64}
    qs::Vector{Float64}
    ns::Vector{Int}
    cum_regret::Vector{Float64}
    plts::Vector{Plots.Plot}
    simple_regret::Vector{Float64}
end
RandomBanditResult() = RandomBanditResult(Float64[], Float64[], Int[], Float64[], Plots.Plot[], Float64[])
function POMDPs.solve(b::RandomBandit, G::ObjectiveFunc, rng::AbstractRNG=MersenneTwister(b.seed))
    actions = Float64[]
    qs = Float64[]
    cum_regret = 0
    g_max = maximum(G) 
    result = RandomBanditResult()
    for n = 1:b.n_iters
        x = next_action(rng, b.actions)
        push!(actions, x)
        r = G(x, rng)
        push!(qs, r)

        if :simple_regret in b.outs
            i = indmax(qs)
            g_best = truth(G, actions[i])
            push!(result.simple_regret, g_max-g_best)
        end
        if :cum_regret in b.outs
            cum_regret += (g_max - r)
            push!(result.cum_regret, cum_regret)
        end
        if :plot in b.outs
            p = plot(G)
            plot!(p, actions, qs, seriestype=:scatter, xlim=(b.actions.xmin,b.actions.xmax), ylim=(0.0,2.0), 
                  title="random")
            push!(result.plts, p)
        end
    end
    append!(result.actions, actions)
    append!(result.qs, qs)
    append!(result.ns, ones(length(actions)))
    result
end
function Plots.animate(result::BanditResult, filename="./result.gif"; fps=5, every=2)
    animate(result.plts, filename, fps=fps, every=every)
end

@with_kw mutable struct PWUCB <: Bandit
    actions::ActionFunc     = UniformActions()
    seed::Int               = 0
    n_iters::Int            = 100
    k::Float64              = 0.5
    α::Float64              = 0.85
    ec::Float64             = 0.5
    q0::Float64             = 0
    n0::Float64             = 0
    outs::Set{Symbol}       = Set{Symbol}()
end
metadata(b::PWUCB) = (:algorithm=>"PW", :seed=>b.seed, :n_iters=>b.n_iters, :k=>b.k, :α=>b.α, :ec=>b.ec, 
                       :q0=>b.q0, :n0=>b.n0)
Base.string(::Type{PWUCB}) = "PW"
struct PWUCBResult <: BanditResult
    actions::Vector{Float64}
    qs::Vector{Float64}
    ns::Vector{Int}
    cum_regret::Vector{Float64}
    plts::Vector{Plots.Plot}
    simple_regret::Vector{Float64}
end
PWUCBResult() = PWUCBResult(Float64[], Float64[], Int[], Float64[], Plots.Plot[], Float64[])
function POMDPs.solve(b::PWUCB, G::ObjectiveFunc, rng::AbstractRNG=MersenneTwister(b.seed))
    actions = Float64[]
    qs = Float64[]
    ns = Float64[]
    result = PWUCBResult()
    cum_regret = 0
    g_max = maximum(G) 
    for n = 1:b.n_iters
        if length(actions) < b.k*n^b.α 
            a = next_action(rng, b.actions)
            push!(actions, a)
            push!(qs, b.q0)
            push!(ns, b.n0)
        end
        ebs = b.ec .* [sqrt(log(n)/na) for na in ns] 
        ebs[isnan.(ebs)] = 0
        ucb_vals = [q+eb for (q,eb) in zip(qs,ebs)]

        i = indmax(ucb_vals)
        x = actions[i]
        r = G(x, rng)
        ns[i] += 1
        qs[i] += (r - qs[i]) / ns[i]

        if :simple_regret in b.outs
            i = indmax(qs)
            g_best = truth(G, actions[i])
            push!(result.simple_regret, g_max-g_best)
        end
        if :cum_regret in b.outs
            cum_regret += (g_max - r)
            push!(result.cum_regret, cum_regret)
        end
        if :plot in b.outs
            p = plot(G)
            plot!(p, actions, qs, err=ebs, seriestype=:scatter, xlim=(b.actions.xmin,b.actions.xmax), 
                  ylim=(0.0,2.0), title="PW")
            push!(result.plts, p)
        end
    end
    append!(result.actions, actions)
    append!(result.qs, qs)
    append!(result.ns, ones(length(actions)))
    result
end

@with_kw mutable struct SBUCB <: Bandit
    actions::ActionFunc     = UniformActions()
    seed::Int               = 0
    n_iters::Int            = 100
    k::Float64              = 0.5
    α::Float64              = 0.5
    ec::Float64             = 0.5
    q0::Float64             = 0
    n0::Float64             = 0
    outs::Set{Symbol}       = Set{Symbol}()
end
metadata(b::SBUCB) = (:algorithm=>"SB", :seed=>b.seed, :n_iters=>b.n_iters, :k=>b.k, :α=>b.α, :ec=>b.ec, 
                       :q0=>b.q0, :n0=>b.n0)
Base.string(::Type{SBUCB}) = "SB"
struct SBUCBResult <: BanditResult
    actions::Vector{Float64}
    qs::Vector{Float64}
    ns::Vector{Int}
    cum_regret::Vector{Float64}
    plts::Vector{Plots.Plot}
    simple_regret::Vector{Float64}
end
SBUCBResult() = SBUCBResult(Float64[], Float64[], Int[], Float64[], Plots.Plot[], Float64[])
function POMDPs.solve(b::SBUCB, G::ObjectiveFunc, rng::AbstractRNG=MersenneTwister(b.seed))
    actions = Float64[]
    qs = Float64[]
    ns = Float64[]
    radii = Float64[]
    result = SBUCBResult()
    cum_regret = 0
    g_max = maximum(G) 
    for n = 1:b.n_iters
        a = next_action(rng, b.actions)
        if isempty(actions)
            d_nn = Inf
        else
            d_nn,i_nn = findmin(dist.(a,actions))
        end
        if d_nn > b.k/n^b.α
            push!(actions, a)
            push!(qs, b.q0)
            push!(ns, b.n0)
            push!(radii, 0)
        end #else, discard
        ebs = b.ec .* [sqrt(log(n)/na) for na in ns] 
        ebs[isnan.(ebs)] = 0
        ucb_vals = [q+eb for (q,eb) in zip(qs,ebs)]

        i = indmax(ucb_vals)
        x = actions[i]
        r = G(x, rng)
        ns[i] += 1
        qs[i] += (r - qs[i]) / ns[i]

        if :cum_regret in b.outs
            cum_regret += (g_max - r)
            push!(result.cum_regret, cum_regret)
        end
        if :plot in b.outs
            fill!(radii, b.k/n^b.α)
            p = plot(G)
            plot!(p, actions, qs, err=ebs, seriestype=:scatter, xlim=(b.actions.xmin,b.actions.xmax), 
                  ylim=(0.0,2.0), title="SB")
            plot!(p, [circle(actions[i],qs[i],radii[i]) for i=1:length(actions)], fillalpha=0.25)
            push!(result.plts, p)
        end
        if :simple_regret in b.outs
            i = indmax(qs)
            g_best = truth(G, actions[i])
            push!(result.simple_regret, g_max-g_best)
        end
    end
    append!(result.actions, actions)
    append!(result.qs, qs)
    append!(result.ns, ones(length(actions)))
    result
end
dist(x::Float64, y::Float64) = norm(x-y)
