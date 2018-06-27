abstract type ObjectiveFunc end

#Test functions
struct DistributionFunc <: ObjectiveFunc
    d::Distribution
    σ::Float64
end
function NarrowBump(d=Normal(-0.25,0.05); σ=0.3)
    DistributionFunc(d, σ)
end
function WideBump(d=Normal(0.15,0.25); σ=0.3)
    DistributionFunc(d, σ)
end
function TwoBumps(d=MixtureModel([Normal(-0.5,0.12), Normal(0.1,0.3)], 
                                  [0.35,0.65]); σ=0.3)
    DistributionFunc(d, σ)
end
function ThreeBumps(d=MixtureModel([Normal(-0.5,0.15),Normal(0.0,0.3),Normal(0.5,0.07)],
                                   [0.27,0.57,0.16]); σ=0.3)
    DistributionFunc(d, σ)
end
function (o::DistributionFunc)(x, rng::AbstractRNG=Base.GLOBAL_RNG)
    pdf(o.d, x) + o.σ*randn(rng)
end
function truth(o::DistributionFunc, x)
    pdf(o.d, x) 
end
struct SincFunc <: ObjectiveFunc
    a::Float64
    b::Float64
    c::Float64
    σ::Float64
end
function Sinc(; a=20, b=0.3, c=0.5, σ=0.3)
    SincFunc(a,b,c,σ)
end
function (o::SincFunc)(x, rng::AbstractRNG=Base.GLOBAL_RNG)
    sinc(o.a*(x-o.b)+o.c) + o.σ*randn(rng)
end
function truth(o::SincFunc, x)
    sinc(o.a*(x-o.b)+o.c) 
end
function circle(x,y,r)
    c = Shape(Plots.partialcircle(0,2π,20,r))
    translate!(c, x, y)
    c
end
function Plots.plot(o::ObjectiveFunc; xmin=-1.0, xmax=1.0, n=100)
    xs = linspace(xmin, xmax, n)
    plot(xs, truth.(o, xs))
end
function Plots.plot(o::ObjectiveFunc, rng::AbstractRNG; xmin=-1.0, xmax=1.0, n=100)
    xs = linspace(xmin, xmax, n)
    plot(xs, o.(xs, rng))
end

function Base.maximum(o::ObjectiveFunc; xmin=-1.0, xmax=1.0, n=1000)
    xs = linspace(xmin, xmax, n)
    g_max = maximum(truth.(o, xs))
    g_max
end
