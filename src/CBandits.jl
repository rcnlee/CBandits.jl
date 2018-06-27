"""
A package for bandit algorithms in 1D continuous space
"""
module CBandits

export ObjectiveFunc, DistributionFunc, SincFunc
export NarrowBump, WideBump, TwoBumps, ThreeBumps, Sinc
export DistributionActions, UniformActions, GaussianActions
export RandomBandit, RandomBanditResult, PWUCB, PWUCBResult, SBUCB, SBUCBResult
export metadata
export BanditSim, MetricStudy, MetricStudyResult, generate_sim_q, run_study

using POMDPs, POMDPToolbox
using Distributions, StatsBase
using Plots; pyplot()
using Parameters

include("objectives.jl")
include("action_dists.jl")
include("algorithms.jl")
include("sim.jl")
include("studies.jl")


end # module
