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
    actions::Vector
    qs::Vector{Float64}
    ns::Vector{Int}
    cum_regret::Vector{Float64}
    plts::Vector{Plots.Plot}
    simple_regret::Vector{Float64}
end
RandomBanditResult() = RandomBanditResult([], Float64[], Int[], Float64[], Plots.Plot[], Float64[])
function POMDPs.solve(b::RandomBandit, G::ObjectiveFunc, rng::AbstractRNG=MersenneTwister(b.seed))
    actions = []
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
            p = plot(G, b, actions, qs)
            push!(result.plts, p)
        end
    end
    append!(result.actions, actions)
    append!(result.qs, qs)
    append!(result.ns, ones(length(actions)))
    result
end
@recipe function plot(G::Objective1D, b::RandomBandit, actions, qs)
    title := string(b)
    ylim := (G.g_min - 1.0, G.g_max + 1.0) 
    @series begin
        G
    end
    @series begin
        seriestype := :scatter
        actions, qs
    end
end
@recipe function plot(G::Objective2D, b::RandomBandit, actions, qs)
    title := string(b)
    zlim := (G.g_min - 1.0, G.g_max + 1.0) 
    @series begin
        G
    end
    @series begin
        seriestype := :scatter
        zcolor := qs
        xs = map(x->x[1], actions)
        ys = map(x->x[2], actions)
        xs, ys 
    end
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
    actions::Vector
    qs::Vector{Float64}
    ns::Vector{Int}
    cum_regret::Vector{Float64}
    plts::Vector{Plots.Plot}
    simple_regret::Vector{Float64}
end
PWUCBResult() = PWUCBResult([], Float64[], Int[], Float64[], Plots.Plot[], Float64[])
function POMDPs.solve(b::PWUCB, G::ObjectiveFunc, rng::AbstractRNG=MersenneTwister(b.seed))
    actions = []
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
            p = plot(G, b, actions, qs, ebs)
            push!(result.plts, p)
        end
    end
    append!(result.actions, actions)
    append!(result.qs, qs)
    append!(result.ns, ones(length(actions)))
    result
end
@recipe function plot(G::Objective1D, b::PWUCB, actions, qs, errbars)
    title := string(b)
    ylim := (G.g_min - 1.0, G.g_max + 1.0) 
    @series begin
        G
    end
    @series begin
        seriestype := :scatter
        title := string(b)
        err := errbars
        actions, qs
    end
end
@recipe function plot(G::Objective2D, b::PWUCB, actions, qs, errbars)
    title := string(b)
    zlim := (G.g_min - 1.0, G.g_max + 1.0) 
    @series begin
        G
    end
    @series begin
        seriestype := :scatter
        #err := errbars  #Figure out how to plot this...
        zcolor := qs
        xs = map(x->x[1], actions)
        ys = map(x->x[2], actions)
        xs, ys 
    end
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
    actions::Vector
    qs::Vector{Float64}
    ns::Vector{Int}
    cum_regret::Vector{Float64}
    plts::Vector{Plots.Plot}
    simple_regret::Vector{Float64}
end
SBUCBResult() = SBUCBResult([], Float64[], Int[], Float64[], Plots.Plot[], Float64[])
function POMDPs.solve(b::SBUCB, G::ObjectiveFunc, rng::AbstractRNG=MersenneTwister(b.seed))
    actions = []
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
            d_nn,i_nn = findmin(dist(a,aa) for aa in actions)
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
            p = plot(G, b, actions, qs, ebs, radii)
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
dist(x, y) = norm(x-y)
@recipe function plot(G::Objective1D, b::SBUCB, actions, qs, errbars, radii)
    title := string(b)
    ylim := (G.g_min - 1.0, G.g_max + 1.0) 
    @series begin
        G
    end
    @series begin
        seriestype := :scatter
        err := errbars
        actions, qs
    end
    @series begin
        fillalpha := 0.25
        [circle(actions[i],qs[i],radii[i]) for i=1:length(actions)]
    end
end
@recipe function plot(G::Objective2D, b::SBUCB, actions, qs, errbars, radii)
    title := string(b)
    zlim := (G.g_min - 1.0, G.g_max + 1.0) 
    @series begin
        G
    end
    @series begin
        seriestype := :scatter
        #err := errbars  #TODO: how to plot this?
        zcolor := qs
        xs = map(x->x[1], actions)
        ys = map(x->x[2], actions)
        xs, ys 
    end
    @series begin
        fillalpha := 0.25
        [circle(actions[i]...,radii[i]) for i=1:length(actions)]
    end
end


#See Srinivas2010 and Dorrard2009
@with_kw mutable struct GPUCBGrid <: Bandit
    actiondistr::ActionDistr = UniformActions()
    grid                     = GridActions()
    seed::Int               = 0
    n_iters::Int            = 100
    n_sig::Float64          = 2.0  #number of stdev's to add to mean (optimism)
    mean_init::Float64      = 0.0
    log_length_scale        = -2.6
    log_signal_sigma        = 0.0
    log_obs_noise           = -1.0
    outs::Set{Symbol}       = Set{Symbol}()
end
Base.string(b::GPUCBGrid) = "GPUCB-Grid"
Base.string(b::Type{GPUCBGrid}) = "GPUCB-Grid"
metadata(b::GPUCBGrid) = [:algorithm=>string(b), :seed=>b.seed, :n_iters=>b.n_iters, :n_sig=>b.n_sig, 
                          :mean_init=>b.mean_init, :log_length_scale=>b.log_length_scale, 
                          :log_signal_sigma=>b.log_signal_sigma, 
                          :log_obs_noise=>b.log_obs_noise]
struct GPUCBGridResult <: BanditResult
    actions::Vector
    qs::Vector{Float64}
    ns::Vector{Int}
    cum_regret::Vector{Float64}
    plts::Vector{Plots.Plot}
    simple_regret::Vector{Float64}
end
GPUCBGridResult() = GPUCBGridResult([], Float64[], Int[], Float64[], Plots.Plot[], Float64[])
function POMDPs.solve(b::GPUCBGrid, G::ObjectiveFunc, rng::AbstractRNG=MersenneTwister(b.seed))
    actions = initial_actions(b.actiondistr) 
    qs = Float64[]
    result = GPUCBGridResult()
    cum_regret = 0
    g_max = maximum(G) 
    gridpoints = grid_points(b.grid)
    mconst = MeanConst(b.mean_init) 
    kern = SE(b.log_length_scale,b.log_signal_sigma)
    gp = GP(vec2mat(actions), qs, mconst, kern, b.log_obs_noise) 
    if :plot in b.outs #plot needs these vars initialized
        m, Σ = predict_f(gp, gridpoints)
        ucb = b.n_sig*sqrt.(Σ)
        ucbmax,i = findmax(m + ucb)
    end
    for n = 1:b.n_iters
        if n == 1
            a = rand(rng, b.actiondistr)
            push!(actions, a)
            imax = 1
        else
            GaussianProcesses.fit!(gp, vec2mat(actions), qs)  
            #optimize!(gp; method=Optim.BFGS())
            m, Σ = predict_f(gp, gridpoints)
            ucb = b.n_sig*sqrt.(Σ)
            ucbmax,imax = findmax(m + ucb)
            push!(actions, getindex_mat(gridpoints,imax))
        end
        x = actions[end] 
        r = G(x, rng)
        push!(qs, r)

        if :cum_regret in b.outs
            cum_regret += (g_max - r)
            push!(result.cum_regret, cum_regret)
        end
        if :plot in b.outs
            p = plot(G, b, gridpoints, m, actions, qs, ucb, ucbmax)
            push!(result.plts, p)
        end
        if :simple_regret in b.outs
            if n == 1
                xbest = x 
            else
                i = indmax(m)
                xbest = getindex_mat(gridpoints,i)
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
initial_actions(A::DistributionActions) = Float64[] 
initial_actions(A::Union{UniformActions2D,DistributionActions2D}) = Vector{Float64}[]
vec2mat(xs::Vector{Float64}) = xs[:,1:1]'
vec2mat(xs::Vector{Vector{Float64}}) = isempty(xs) ? Array{Float64,2}(2,0) : hcat(xs...)
getindex_mat(xs::Vector{Float64},i) = xs[i]
getindex_mat(xs::Vector{Vector{Float64}},i) = xs[i]
getindex_mat(xs::Array{Float64,2},i) = xs[:,i]
@recipe function plot(G::Objective1D, b::GPUCBGrid, gridpoints, m, actions, qs, ucb, ucbmax)
    title := string(b)
    ylim := (G.g_min - 1.0, G.g_max + 1.0) 
    @series begin
        G
    end
    @series begin
        seriestype := :scatter
        label := "observations"
        actions[1:end-1], qs[1:end-1]
    end
    @series begin
        seriestype := :path
        linestyle := :dash
        ribbon := ucb
        label := "predicted mean"
        gridpoints, m
    end
    @series begin
        seriestype := :scatter
        markershape := :star4
        label := "ucb max"
        actions[end:end], [ucbmax]
    end
end
@recipe function plot(G::Objective2D, b::GPUCBGrid, gridpoints, m, actions, qs, ucb, ucbmax)
    title := string(b)
    zlim := (G.g_min - 1.0, G.g_max + 1.0) 
    layout := @layout grid(1,3) 
    @series begin
        subplot := 1
        G
    end
    @series begin
        subplot := 1
        seriestype := :scatter
        label := "observations"
        xs = map(x->x[1], actions[1:end-1])
        ys = map(x->x[2], actions[1:end-1])
        xs, ys
    end
    @series begin
        subplot := 2
        seriestype := :heatmap
        title := "predicted mean"
        legend := true 
        b.grid.xs, b.grid.ys, reshape(m, (b.grid.n,b.grid.n))'
    end
    @series begin
        subplot := 3
        seriestype := :heatmap
        title := "ucb"
        legend := true 
        b.grid.xs, b.grid.ys, reshape(ucb, (b.grid.n,b.grid.n))'
    end
end

@with_kw mutable struct GPUCB <: Bandit
    actiondistr::ActionDistr  = UniformActions()
    grid                     = GridActions()  #for plotting only
    seed::Int               = 0
    n_iters::Int            = 100
    k::Int                  = 10
    n_sig::Float64          = 2.0  #number of stdev's to add to mean (optimism)
    mean_init::Float64      = 0.0
    log_length_scale        = -2.6
    log_signal_sigma        = 0.0
    log_obs_noise           = -1.0
    outs::Set{Symbol}       = Set{Symbol}()
end
Base.string(b::GPUCB) = "GPUCB"
Base.string(b::Type{GPUCB}) = "GPUCB"
metadata(b::GPUCB) = [:algorithm=>string(b), :seed=>b.seed, :n_iters=>b.n_iters, :k=>b.k, :n_sig=>b.n_sig,
                      :mean_init=>b.mean_init, :log_length_scale=>b.log_length_scale, 
                      :log_signal_sigma=>b.log_signal_sigma, 
                      :log_obs_noise=>b.log_obs_noise]
struct GPUCBResult <: BanditResult
    actions::Vector
    qs::Vector{Float64}
    ns::Vector{Int}
    cum_regret::Vector{Float64}
    plts::Vector{Plots.Plot}
    simple_regret::Vector{Float64}
end
GPUCBResult() = GPUCBResult([], Float64[], Int[], Float64[], Plots.Plot[], Float64[])
function POMDPs.solve(b::GPUCB, G::ObjectiveFunc, rng::AbstractRNG=MersenneTwister(b.seed))
    actions = initial_actions(b.actiondistr) 
    qs = Float64[]
    result = GPUCBResult()
    cum_regret = 0
    g_max = maximum(G) 
    gridpoints = grid_points(b.grid) #for plotitng only
    mconst = MeanConst(b.mean_init) 
    kern = SE(b.log_length_scale,b.log_signal_sigma)
    gp = GP(vec2mat(actions), qs, mconst, kern, b.log_obs_noise) 
    for n = 1:b.n_iters
        if n == 1
            a = rand(rng, b.actiondistr)
            push!(actions, a)
            imax = 1
        else
            GaussianProcesses.fit!(gp, vec2mat(actions), qs)  #fit over existing actions
            #optimize!(gp; method=Optim.BFGS())
            xs = vcat(actions, [rand(rng, b.actiondistr) for i=1:b.k])
            m, Σ = predict_f(gp, vec2mat(xs))  #over actions and k random samples from proposal distribution
            ucb = b.n_sig*sqrt.(Σ)
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
            if n != 1
                m´, Σ´ = predict_f(gp, gridpoints) #over plotting grid
                ucb´ = b.n_sig*sqrt.(Σ´)
                ucbmax´ = maximum(m´ + ucb´)
                p = plot(G, b, gridpoints, m´, actions, qs, ucb´, ucbmax´)
                push!(result.plts, p)
            end
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
@recipe function plot(G::Objective1D, b::GPUCB, xs, m, actions, qs, ucb, ucbmax)
    title := string(b)
    ylim := (G.g_min - 1.0, G.g_max + 1.0) 
    @series begin
        G
    end
    @series begin
        seriestype := :scatter
        label := "observations"
        actions[1:end-1], qs[1:end-1]
    end
    @series begin
        seriestype := :path
        linestyle := :dash
        ribbon := ucb
        label := "predicted mean"
        xs, m
    end
    @series begin
        seriestype := :scatter
        markershape := :star4
        label := "ucb max"
        actions[end:end], [ucbmax]
    end
end
@recipe function plot(G::Objective2D, b::GPUCB, gridpoints, m, actions, qs, ucb, ucbmax)
    title := string(b)
    zlim := (G.g_min - 1.0, G.g_max + 1.0) 
    layout := @layout grid(1,3) 
    @series begin
        subplot := 1
        G
    end
    @series begin
        subplot := 1
        seriestype := :scatter
        label := "observations"
        xs = map(x->x[1], actions[1:end-1])
        ys = map(x->x[2], actions[1:end-1])
        xs, ys
    end
    @series begin
        subplot := 2
        seriestype := :heatmap
        title := "predicted mean"
        legend := true 
        b.grid.xs, b.grid.ys, reshape(m, (b.grid.n,b.grid.n))'
    end
    @series begin
        subplot := 3
        title := "ucb"
        legend := true 
        b.grid.xs, b.grid.ys, reshape(ucb, (b.grid.n,b.grid.n))'
    end
end
