#Action spaces
abstract type ActionDistr end
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
function Plots.plot(o::DistributionActions; xmin=-1.0, xmax=1.0, n=100)
    xs = linspace(xmin, xmax, n)
    ys = pdf.(o.d, xs)
    plot(xs, ys, xlim=(xmin,xmax), ylim=(0.0,maximum(ys)+0.5))
end
