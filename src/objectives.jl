abstract type ObjectiveFunc end
abstract type Objective1D  <: ObjectiveFunc end
abstract type Objective2D  <: ObjectiveFunc end

#Test functions
struct DistributionFunc <: Objective1D
    name::String
    d::Distribution
    σ::Float64
    g_max::Float64
    x_max::Float64
end
function DistributionFunc(name::String, d::Distribution, σ::Float64; xmin=-1.0, xmax=1.0, n=1000)
    xs = linspace(xmin, xmax, n)
    g_max, i_max = findmax(pdf.(d, xs))
    DistributionFunc(name, d, σ, g_max, xs[i_max])
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
function (o::DistributionFunc)(x, rng::AbstractRNG=Base.GLOBAL_RNG)
    pdf(o.d, x) + o.σ*randn(rng)
end
function truth(o::DistributionFunc, x)
    pdf(o.d, x) 
end
Base.string(o::ObjectiveFunc) = o.name
struct SincFunc <: Objective1D
    name::String
    a::Float64
    b::Float64
    c::Float64
    σ::Float64
    g_max::Float64
    x_max::Float64
end
function SincFunc(name::String, a::Float64, b::Float64, c::Float64, σ::Float64; xmin=-1.0, xmax=1.0, n=1000)
    xs = linspace(xmin, xmax, n)
    g_max, i_max = findmax(sinc_truth.(a, b, c, xs))
    SincFunc(name, a, b, c, σ, g_max, xs[i_max])
end
function Sinc(; a=20.0, b=0.3, c=0.5, σ=0.3)
    SincFunc("Sinc", a,b,c,σ)
end
function (o::SincFunc)(x, rng::AbstractRNG=Base.GLOBAL_RNG)
    truth(o,x) + o.σ*randn(rng)
end
truth(o::SincFunc, x) = sinc_truth(o.a, o.b, o.c, x)
sinc_truth(a::Float64, b::Float64, c::Float64, x::Float64) = sinc(a * (x-b) + c) 
function circle(x,y,r)
    c = Shape(Plots.partialcircle(0,2π,20,r))
    translate!(c, x, y)
    c
end
@recipe function f(o::Objective1D; xmin=-1.0, xmax=1.0, n=100)
    @series begin
        seriestype := :path
        xlabel := "x"
        ylabel := "f(x)"
        label := "true mean"
        xs = linspace(xmin, xmax, n)
        xs, truth.(o, xs)
    end
end
@recipe function f(o::Objective1D, rng::AbstractRNG; xmin=-1.0, xmax=1.0, n=100)
    @series begin
        seriestype := :path
        xlabel := "x"
        ylabel := "f(x)"
        label := "true mean"
        xs = linspace(xmin, xmax, n)
        xs, o.(xs, rng)
    end
end
@recipe function f(o::Objective2D; xmin=-1.0, xmax=1.0, n=100)
    @assert false
    @series begin
        seriestype := :path
        xlabel := "x"
        ylabel := "f(x)"
        label := "true mean"
        xs = linspace(xmin, xmax, n)
        xs, truth.(o, xs)
    end
end
@recipe function f(o::Objective2D, rng::AbstractRNG; xmin=-1.0, xmax=1.0, n=100)
    @assert false
    @series begin
        seriestype := :path
        xlabel := "x"
        ylabel := "f(x)"
        label := "true mean"
        xs = linspace(xmin, xmax, n)
        xs, o.(xs, rng)
    end
end

Base.maximum(o::Objective1D) = o.g_max
