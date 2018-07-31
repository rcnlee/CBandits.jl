
module GPPerformance

export run_study
export GPNvsTime, GPNvsTimeResult
export GP2NvsTime, GP2NvsTimeResult

using GaussianProcesses, GaussianProcesses2
using CPUTime, Plots, DataFrames
using Parameters

@with_kw struct GPNvsTime
    ds::Vector{Int} = [2,3]
    ns::Vector{Int} = [10,20,50,100,200,500,700,1000,1500,2000] 
    seeds::Vector{Int} = collect(1:5)
end

struct GPNvsTimeResult
    data::DataFrame
end

function run_study(study::GPNvsTime)
    mzero = MeanZero()
    kern = SE(0.0,0.0)
    v = 0.02
    logobsnoise = log(sqrt(v)) 

    ####
    # make sure things are compiled before timing
    X = rand(2, 5)
    y = rand(5) 
    gp = GP(X, y, mzero, kern, logobsnoise)
    predict_f(gp, X)
    ####

    df = DataFrame([Int,Int,Int,Float64],[:dims,:n,:seed,:cputime_us],0)
    for d in study.ds, n in study.ns, s in study.seeds
        rng = MersenneTwister(s)
        X = rand(rng, d, n)
        y = rand(rng, n) 
        tstart = CPUtime_us()
        gp = GP(X, y, mzero, kern, logobsnoise)
        predict_f(gp, X)
        t = CPUtime_us() - tstart
        push!(df, [d, n, s, t])
    end
    GPNvsTimeResult(df)
end

@with_kw struct GP2NvsTime
    ds::Vector{Int} = [2,3]
    ns::Vector{Int} = [10,20,50,100,200,500,700,1000,1500,2000] 
    seeds::Vector{Int} = collect(1:5)
end

struct GP2NvsTimeResult
    data::DataFrame
end

function run_study(study::GP2NvsTime)
    mzero = ZeroMean()
    kern = SquaredExponential(1.0) 
    v = 0.02

    ####
    # make sure things are compiled before timing
    X = [rand(2) for _ = 1:5]
    y = rand(5) 
    gp = GaussianProcess(mzero, kern, X, y, v)
    GaussianProcesses2.predict(gp, X)
    ####

    df = DataFrame([Int,Int,Int,Float64],[:dims,:n,:seed,:cputime_us],0)
    for d in study.ds, n in study.ns, s in study.seeds
        rng = MersenneTwister(s)
        X = [rand(rng, d) for _ in 1:n]
        y = rand(rng, n) 
        tstart = CPUtime_us()
        gp = GaussianProcess(mzero, kern, X, y, v)
        GaussianProcesses2.predict(gp, X)
        t = CPUtime_us() - tstart
        push!(df, [d, n, s, t])
    end
    GP2NvsTimeResult(df)
end

@recipe function plot(result::Union{GPNvsTimeResult,GP2NvsTimeResult})
    xlabel := "number of points"
    ylabel := "cpu time (us)"
    df = aggregate(result.data, [:dims,:n], [mean,std,length])
    for dd in groupby(df, :dims)
        @series begin
            seriestype := :path
            err := dd[:cputime_us_std] / dd[1,:seed_length]
            label := "$(string(result))-$(dd[1,:dims])d"
            dd[:n], dd[:cputime_us_mean]
        end
    end
end
string(result::GPNvsTimeResult) = "GP1"
string(result::GP2NvsTimeResult) = "GP2"

@recipe function plot(result1::GPNvsTimeResult, result2::GP2NvsTimeResult)
   @series begin
       result1
   end
   @series begin
       result2
   end
end

end
