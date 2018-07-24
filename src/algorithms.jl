#Algorithms
abstract type Bandit end
abstract type BanditResult end
@with_kw mutable struct RandomBandit <: Bandit
    actiondistr::ActionDistr  = UniformActions()
    seed::Int               = 0
    n_iters::Int            = 100
    outs::Set{Symbol}       = Set{Symbol}()
end
metadata(x) = []
Base.string(b::RandomBandit) = "Random"
Base.string(b::Type{RandomBandit}) = "Random"
metadata(b::RandomBandit) = [:algorithm=>string(b), :seed=>b.seed, :n_iters=>b.n_iters]
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
        x = rand(rng, b.actiondistr)
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
            plot!(p, actions, qs, seriestype=:scatter, xlim=(b.actiondistr.xmin,b.actiondistr.xmax), 
                  ylim=(-1.0,g_max+1.0), title=string(b))
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
    actiondistr::ActionDistr  = UniformActions()
    seed::Int               = 0
    n_iters::Int            = 100
    k::Float64              = 0.5
    α::Float64              = 0.85
    ec::Float64             = 0.5
    q0::Float64             = 0
    n0::Float64             = 0
    outs::Set{Symbol}       = Set{Symbol}()
end
metadata(b::PWUCB) = [:algorithm=>"PW", :seed=>b.seed, :n_iters=>b.n_iters, :k=>b.k, :α=>b.α, :ec=>b.ec, 
                       :q0=>b.q0, :n0=>b.n0]
Base.string(b::PWUCB) = "PW"
Base.string(b::Type{PWUCB}) = "PW"
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
            a = rand(rng, b.actiondistr)
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
            plot!(p, actions, qs, err=ebs, seriestype=:scatter, xlim=(b.actiondistr.xmin,b.actiondistr.xmax), 
                  ylim=(-1.0,g_max+1.0), title=string(b))
            push!(result.plts, p)
        end
    end
    append!(result.actions, actions)
    append!(result.qs, qs)
    append!(result.ns, ones(length(actions)))
    result
end

@with_kw mutable struct SBUCB <: Bandit
    actiondistr::ActionDistr  = UniformActions()
    seed::Int               = 0
    n_iters::Int            = 100
    k::Float64              = 0.5
    α::Float64              = 0.5
    ec::Float64             = 0.5
    q0::Float64             = 0
    n0::Float64             = 0
    outs::Set{Symbol}       = Set{Symbol}()
end
Base.string(b::SBUCB) = "SB"
Base.string(b::Type{SBUCB}) = "SB"
metadata(b::SBUCB) = [:algorithm=>string(b), :seed=>b.seed, :n_iters=>b.n_iters, :k=>b.k, :α=>b.α, :ec=>b.ec, 
                       :q0=>b.q0, :n0=>b.n0]
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
        a = rand(rng, b.actiondistr)
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
            plot!(p, actions, qs, err=ebs, seriestype=:scatter, xlim=(b.actiondistr.xmin,b.actiondistr.xmax), 
                  ylim=(-1.0,g_max+1.0), title=string(b))
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


#See Srinivas2010 and Dorrard2009
@with_kw mutable struct GPUCBGrid <: Bandit
    actiondistr::ActionDistr  = UniformActions()
    seed::Int               = 0
    n_iters::Int            = 100
    sqrtβ::Float64          = 2.0  #number of stdev's to add to mean (optimism)
    log_length_scale        = -2.3
    log_signal_sigma        = 0.0
    log_obs_noise           = -2.0
    outs::Set{Symbol}       = Set{Symbol}()
end
Base.string(b::GPUCBGrid) = "GPUCB-Grid"
Base.string(b::Type{GPUCBGrid}) = "GPUCB-Grid"
metadata(b::GPUCBGrid) = [:algorithm=>string(b), :seed=>b.seed, :n_iters=>b.n_iters, :sqrtβ=>b.sqrtβ, 
                          :log_length_scale=>b.log_length_scale, :log_signal_sigma=>b.log_signal_sigma, 
                          :log_obs_noise=>b.log_obs_noise]
struct GPUCBGridResult <: BanditResult
    actions::Vector{Float64}
    qs::Vector{Float64}
    ns::Vector{Int}
    cum_regret::Vector{Float64}
    plts::Vector{Plots.Plot}
    simple_regret::Vector{Float64}
end
GPUCBGridResult() = GPUCBGridResult(Float64[], Float64[], Int[], Float64[], Plots.Plot[], Float64[])
function POMDPs.solve(b::GPUCBGrid, G::ObjectiveFunc, rng::AbstractRNG=MersenneTwister(b.seed))
    actions = Float64[]
    qs = Float64[]
    result = GPUCBGridResult()
    cum_regret = 0
    g_max = maximum(G) 
    xs = linspace(b.actiondistr.xmin,b.actiondistr.xmax,100)
    mZero = MeanZero()
    kern = SE(b.log_length_scale,b.log_signal_sigma)
    gp = GP(actions, qs, mZero, kern, b.log_obs_noise) 
    if :plot in b.outs #plot needs these vars initialized
        m,v = predict_y(gp, xs)  
        ucb = b.sqrtβ*sqrt.(v)
        ucbmax,i = findmax(m + ucb)
    end
    for n = 1:b.n_iters
        if n == 1
            a = rand(rng, b.actiondistr)
            push!(actions, a)
            imax = 1
        else
            ScikitLearnBase.fit!(gp, actions[:,1:1], qs)
            #optimize!(gp; method=Optim.BFGS())
            m,v = predict_y(gp, xs)
            ucb = b.sqrtβ*sqrt.(v)
            ucbmax,imax = findmax(m + ucb)
            push!(actions, xs[imax])
        end
        x = actions[end] 
        r = G(x, rng)
        push!(qs, r)

        if :cum_regret in b.outs
            cum_regret += (g_max - r)
            push!(result.cum_regret, cum_regret)
        end
        if :plot in b.outs
            p = plot(G)
            plot!(p, actions[1:end-1], qs[1:end-1], seriestype=:scatter, xlim=(b.actiondistr.xmin,b.actiondistr.xmax), 
                  ylim=(-1.0,g_max+1.0), title=string(b))
            plot!(p, xs, m; seriestype=:path, linestyle=:dash, ribbon=ucb)
            plot!(p, actions[end:end], [ucbmax], seriestype=:scatter, markershape=:star4, 
                  xlim=(b.actiondistr.xmin,b.actiondistr.xmax), 
                  ylim=(-1.0,g_max+1.0), title=string(b))
            push!(result.plts, p)
        end
        if :simple_regret in b.outs
            if n == 1
                xbest = x 
            else
                i = indmax(m)
                xbest = xs[i]
            end
            g_best = truth(G, xbest)
            push!(result.simple_regret, g_max-g_best)
        end
    end
    append!(result.actions, actions)
    append!(result.qs, qs)
    append!(result.ns, ones(length(actions)))
    result
end

@with_kw mutable struct GPUCB <: Bandit
    actiondistr::ActionDistr  = UniformActions()
    seed::Int               = 0
    n_iters::Int            = 100
    k::Int                  = 50
    sqrtβ::Float64          = 2.0  #number of stdev's to add to mean (optimism)
    log_length_scale        = -2.3
    log_signal_sigma        = 0.0
    log_obs_noise           = -2.0
    outs::Set{Symbol}       = Set{Symbol}()
end
Base.string(b::GPUCB) = "GPUCB"
Base.string(b::Type{GPUCB}) = "GPUCB"
metadata(b::GPUCB) = [:algorithm=>string(b), :seed=>b.seed, :n_iters=>b.n_iters, :k=>b.k, :sqrtβ=>b.sqrtβ]
struct GPUCBResult <: BanditResult
    actions::Vector{Float64}
    qs::Vector{Float64}
    ns::Vector{Int}
    cum_regret::Vector{Float64}
    plts::Vector{Plots.Plot}
    simple_regret::Vector{Float64}
end
GPUCBResult() = GPUCBResult(Float64[], Float64[], Int[], Float64[], Plots.Plot[], Float64[])
function POMDPs.solve(b::GPUCB, G::ObjectiveFunc, rng::AbstractRNG=MersenneTwister(b.seed))
    actions = Float64[]
    qs = Float64[]
    result = GPUCBResult()
    cum_regret = 0
    g_max = maximum(G) 
    plot_xs = linspace(b.actiondistr.xmin,b.actiondistr.xmax,100) #used for plotting only
    mZero = MeanZero()
    kern = SE(b.log_length_scale,b.log_signal_sigma)
    gp = GP(actions, qs, mZero, kern, b.log_obs_noise) 
    if :plot in b.outs #plot needs these vars initialized
        m,v = predict_y(gp, plot_xs)  
        ucb = b.sqrtβ*sqrt.(v)
        ucbmax,i = findmax(m + ucb)
    end
    for n = 1:b.n_iters
        if n == 1
            a = rand(rng, b.actiondistr)
            push!(actions, a)
            imax = 1
        else
            ScikitLearnBase.fit!(gp, actions[:,1:1], qs)
            #optimize!(gp; method=Optim.BFGS())
            xs = vcat(actions, [rand(rng, b.actiondistr) for i=1:b.k])
            m,v = predict_y(gp, xs)
            ucb = b.sqrtβ*sqrt.(v)
            ucbmax,imax = findmax(m + ucb)
            push!(actions, xs[imax])
        end
        x = actions[end] 
        r = G(x, rng)
        push!(qs, r)

        if :cum_regret in b.outs
            cum_regret += (g_max - r)
            push!(result.cum_regret, cum_regret)
        end
        if :plot in b.outs
            p = plot(G)
            plot!(p, actions[1:end-1], qs[1:end-1], seriestype=:scatter, xlim=(b.actiondistr.xmin,b.actiondistr.xmax), 
                  ylim=(-1.0,g_max+1.0), title=string(b))
            if n != 1
                m,v = predict_y(gp, plot_xs)
                ucb = b.sqrtβ*sqrt.(v)
                ucbmax,imax = findmax(m + ucb)
                plot!(p, plot_xs, m; seriestype=:path, linestyle=:dash, ribbon=ucb)
                plot!(p, actions[end:end], [ucbmax], seriestype=:scatter, markershape=:star4, 
                    xlim=(b.actiondistr.xmin,b.actiondistr.xmax), 
                    ylim=(-1.0,g_max+1.0), title=string(b))
            end
            push!(result.plts, p)
        end
        if :simple_regret in b.outs
            if n == 1
                xbest = x 
            else
                i = indmax(m)
                xbest = xs[i]
            end
            g_best = truth(G, xbest)
            push!(result.simple_regret, g_max-g_best)
        end
    end
    append!(result.actions, actions)
    append!(result.qs, qs)
    append!(result.ns, ones(length(actions)))
    result
end
