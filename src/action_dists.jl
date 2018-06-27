#Action spaces
abstract type ActionFunc end
struct DistributionActions <: ActionFunc
    xmin::Float64
    xmax::Float64
    d::Distribution
end
UniformActions(; xmin=-1.0, xmax=1.0) = DistributionActions(xmin, xmax, Uniform(xmin, xmax))
GaussianActions(; xmin=-1.0, xmax=1.0, m=0.0, s=0.25) = DistributionActions(xmin, xmax, Normal(m,s))
next_action(o::DistributionActions) = next_action(Base.GLOBAL_RNG, o)
function next_action(rng::AbstractRNG, o::DistributionActions)
    rand(rng, o.d)
end
function Plots.plot(o::DistributionActions; xmin=-1.0, xmax=1.0, n=100)
    xs = linspace(xmin, xmax, n)
    ys = pdf.(o.d, xs)
    plot(xs, ys, xlim=(xmin,xmax), ylim=(0.0,maximum(ys)+0.5))
end
