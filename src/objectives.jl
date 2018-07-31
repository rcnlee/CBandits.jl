abstract type ObjectiveFunc end
abstract type Objective1D  <: ObjectiveFunc end
abstract type Objective2D  <: ObjectiveFunc end

Base.string(o::ObjectiveFunc) = o.name

#Test functions
mutable struct DistributionFunc <: Objective1D
    name::String
    d::Distribution
    σ::Float64
    xlim::Tuple{Float64,Float64}
    g_max::Float64
    g_min::Float64
    x_max::Float64

    function DistributionFunc(name::String, d::Distribution, σ::Float64; xmin=-1.0, xmax=1.0, n=1000)
        o = new(name, d, σ, (xmin,xmax))
        xs = linspace(xmin, xmax, n)
        o.g_max, i_max = findmax(truth.(o, xs))
        o.x_max = xs[i_max]
        o.g_min = minimum(truth.(o, xs))
        o
    end
end
function NarrowBump(d=Normal(-0.25,0.05); σ=0.3)
    DistributionFunc("NarrowBump", d, σ)
end
function WideBump(d=Normal(0.15,0.25); σ=0.3)
    DistributionFunc("WideBump", d, σ)
end
function TwoBumps(d=MixtureModel([Normal(-0.5,0.12), Normal(0.1,0.3)], 
                                  [0.35,0.65]); σ=0.3)
    DistributionFunc("TwoBumps", d, σ)
end
function ThreeBumps(d=MixtureModel([Normal(-0.5,0.15),Normal(0.0,0.3),Normal(0.5,0.07)],
                                   [0.27,0.57,0.16]); σ=0.3)
    DistributionFunc("ThreeBumps", d, σ)
end
truth(o::DistributionFunc, x) = pdf(o.d, x) 
function (o::DistributionFunc)(x, rng::AbstractRNG=Base.GLOBAL_RNG) 
    truth(o,x) + o.σ*randn(rng)
end

mutable struct SincFunc <: Objective1D
    name::String
    a::Float64
    b::Float64
    c::Float64
    σ::Float64
    xlim::Tuple{Float64,Float64}
    g_max::Float64
    x_max::Float64
    g_min::Float64

    function SincFunc(name::String, a::Float64, b::Float64, c::Float64, σ::Float64; xmin=-1.0, xmax=1.0, n=1000)
        o = new(name, a, b, c, σ, (xmin,xmax))
        xs = linspace(xmin, xmax, n)
        o.g_max, i_max = findmax(truth.(o, xs))
        o.x_max = xs[i_max]
        o.g_min = minimum(truth.(o, xs))
        o
    end
end
function Sinc(; a=20.0, b=0.3, c=0.5, σ=0.3)
    SincFunc("Sinc", a,b,c,σ)
end
sinc_truth(a::Float64, b::Float64, c::Float64, x::Float64) = sinc(a * (x-b) + c) 
truth(o::SincFunc, x) = sinc_truth(o.a, o.b, o.c, x)
function (o::SincFunc)(x, rng::AbstractRNG=Base.GLOBAL_RNG) 
    truth(o,x) + o.σ*randn(rng)
end
function circle(x,y,r)
    c = Shape(Plots.partialcircle(0,2π,20,r))
    translate!(c, x, y)
    c
end
@recipe function plot(o::Objective1D; xmin=-1.0, xmax=1.0, n=100)
    xlim := o.xlim
    ylim := (o.g_min - 1.0, o.g_max + 1.0) 
    xlabel := "x"
    ylabel := "f(x)"
    @series begin
        seriestype := :path
        label := "true mean"
        xs = linspace(xmin, xmax, n)
        xs, truth.(o, xs)
    end
end
@recipe function plot(o::Objective1D, rng::AbstractRNG; xmin=-1.0, xmax=1.0, n=100)
    xlim := o.xlim
    ylim := (o.g_min - 1.0, o.g_max + 1.0) 
    xlabel := "x"
    ylabel := "f(x)"
    @series begin
        seriestype := :path
        label := "noisy sample"
        xs = linspace(xmin, xmax, n)
        xs, o.(xs, rng)
    end
end

Base.maximum(o::Objective1D) = o.g_max
Base.minimum(o::Objective1D) = o.g_min

#####
# 2D
mutable struct DistributionFunc2D <: Objective2D
    name::String
    d::Distribution
    σ::Float64
    xlim::Tuple{Float64,Float64}
    ylim::Tuple{Float64,Float64}
    g_max::Float64
    g_min::Float64
    x_max::Vector{Float64}

    function DistributionFunc2D(name::String, d::Distribution, σ::Float64; 
                                xlim=(-2.0,2.0), ylim=(-2.0,2.0), n=1000)
        o = new(name, d, σ, xlim, ylim)
        Xs = [[x,y] for x in linspace(xlim..., n), y in linspace(ylim..., n)]
        o.g_max, i_max = findmax(truth.(o,Xs))
        o.x_max = Xs[i_max]
        o.g_min = minimum(truth.(o,Xs))
        o
    end
end
function NarrowBump2D(d=MvNormal([0.4,0.4],[0.2,0.2]); σ=2.0)
    DistributionFunc2D("NarrowBump2D", d, σ)
end
function WideBump2D(d=MvNormal([0.25,0.25],[0.2,0.7]); σ=0.5)
    DistributionFunc2D("WideBump2D", d, σ)
end
truth(o::DistributionFunc2D, x) = pdf(o.d, x) 
function (o::DistributionFunc2D)(x, rng::AbstractRNG=Base.GLOBAL_RNG) 
    truth(o,x) + o.σ*randn(rng)
end


mutable struct Rosenbrock <: Objective2D
    name::String
    σ::Float64
    xlim::Tuple{Float64,Float64}
    ylim::Tuple{Float64,Float64}
    g_max::Float64
    g_min::Float64
    x_max::Vector{Float64}

    function Rosenbrock(σ::Float64=10.0; xlim=(-2.0,2.0), ylim=(-2.0,2.0), n=1000)
        o = new("Rosenbrock", σ, xlim, ylim)
        Xs = [[x,y] for x in linspace(xlim..., n), y in linspace(ylim..., n)]
        o.g_max, i_max = findmax(truth.(o,Xs))
        o.x_max = Xs[i_max]
        o.g_min = minimum(truth.(o,Xs))
        o
    end
end
truth(o::Rosenbrock, x) = -rosenbrock(x) 
function (o::Rosenbrock)(x, rng::AbstractRNG=Base.GLOBAL_RNG) 
    truth(o,x) + o.σ*randn(rng)
end

@recipe function plot(o::Objective2D; n=100)
    xlim := o.xlim
    ylim := o.ylim
    zlim := (o.g_min - 1.0, o.g_max + 1.0) 
    xlabel := "x"
    ylabel := "y"
    zlabel := "f(x,y)"
    @series begin
        seriestype := :heatmap
        label := "true mean"
        xs = linspace(o.xlim..., n)
        ys = linspace(o.ylim..., n)
        xs, ys, (x,y)->truth(o, [x,y])
    end
end
@recipe function plot(o::Objective2D, rng::AbstractRNG; n=100)
    xlim := o.xlim
    ylim := o.ylim
    zlim := (o.g_min - 1.0, o.g_max + 1.0) 
    xlabel := "x"
    ylabel := "y"
    zlabel := "f(x,y)"
    @series begin
        seriestype := :heatmap
        label := "noisy sample"
        xs = linspace(o.xlim..., n)
        ys = linspace(o.ylim..., n)
        xs, ys, (x,y)->o([x,y],rng)
    end
end

Base.maximum(o::Objective2D) = o.g_max
Base.minimum(o::Objective2D) = o.g_min
