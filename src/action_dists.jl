#Action spaces
abstract type ActionDistr end

#####
struct DistributionActions <: ActionDistr
    xmin::Float64
    xmax::Float64
    d::Distribution
end
UniformActions(; xmin=-1.0, xmax=1.0) = DistributionActions(xmin, xmax, Uniform(xmin, xmax))
GaussianActions(; xmin=-1.0, xmax=1.0, m=0.0, s=0.25) = DistributionActions(xmin, xmax, Normal(m,s))
Base.rand(o::DistributionActions) = rand(Base.GLOBAL_RNG, o)
function Base.rand(rng::AbstractRNG, o::DistributionActions)
    rand(rng, o.d)
end
@recipe function plot(o::DistributionActions; xmin=-1.0, xmax=1.0, n=100)
    xs = linspace(xmin, xmax, n)
    ys = pdf.(o.d, xs)
    xlim := xmin,xmax 
    ylim := (0.0, maximum(ys)+0.5)
    @series begin
        xs, ys
    end
end

#####
struct UniformActions2D <: ActionDistr
    xlim::Tuple{Float64,Float64}
    ylim::Tuple{Float64,Float64}
    dvec::Vector{Uniform{Float64}}

   function UniformActions2D(; xlim=(-2.0,2.0), ylim=(-2.0,2.0))
        new(xlim, ylim, [Uniform(xlim...), Uniform(ylim...)])
    end
end
Distributions.pdf(o::UniformActions2D, x, y) = 1/((o.xlim[2]-o.xlim[1])*(o.ylim[2]-o.ylim[1]))
Base.rand(o::UniformActions2D) = rand(Base.GLOBAL_RNG, o)
Base.rand(rng::AbstractRNG, o::UniformActions2D) = rand.(rng, o.dvec)
@recipe function plot(o::UniformActions2D; n=100)
    @series begin
        xs = linspace(o.xlim..., n)
        ys = linspace(o.ylim..., n)
        z = (x,y)->pdf(o,x,y)
        xlim := o.xlim 
        ylim := o.ylim
        seriestype := :heatmap
        xs, ys, z
    end
end

struct GridActions 
    xlim::Tuple{Float64,Float64}
    n::Int
    X::Vector{Float64}
end
GridActions() = GridActions((-1.0,1.0), 100)
grid_points(A::GridActions) = A.X

#####
struct DistributionActions2D <: ActionDistr
    xlim::Tuple{Float64,Float64}
    ylim::Tuple{Float64,Float64}
    d::Distribution
end
function GaussianActions2D(m=[0.0,0.0], s=[0.85,0.85]; xlim=(-2.0,2.0), ylim=(-2.0,2.0))
    DistributionActions2D(xlim, ylim, MvNormal(m, s))
end
Base.rand(o::DistributionActions2D) = rand(Base.GLOBAL_RNG, o)
function Base.rand(rng::AbstractRNG, o::DistributionActions2D) 
    r = rand.(rng, o.d)
    r[1] = clamp(r[1], o.xlim...) 
    r[2] = clamp(r[2], o.ylim...) 
    r
end
Distributions.pdf(o::DistributionActions2D, x) = pdf(o.d, x)
@recipe function plot(o::DistributionActions2D; n=20)
    @series begin
        xs = linspace(o.xlim..., n)
        ys = linspace(o.ylim..., n)
        z = (x,y)->pdf(o, [x,y])
        xlim := o.xlim 
        ylim := o.ylim
        seriestype := :heatmap
        xs, ys, z
    end
end

struct GridActions2D 
    xlim::Tuple{Float64,Float64}
    ylim::Tuple{Float64,Float64}
    n::Int
    xs::Vector{Float64}
    ys::Vector{Float64}
    X::Array{Float64,2}
    function GridActions2D(xlim::Tuple{Float64,Float64}, ylim::Tuple{Float64,Float64}, n::Int)
        xs = linspace(xlim..., n) |> collect
        ys = linspace(ylim..., n) |> collect
        X = hcat([[x,y] for x in xs, y in ys]...)
        new(xlim, ylim, n, xs, ys, X)
    end
end
GridActions2D() = GridActions2D((-2.0,2.0), (-2.0,2.0), 100)
grid_points(A::GridActions2D) = A.X
