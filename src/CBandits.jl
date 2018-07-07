"""
A package for bandit algorithms in 1D continuous space
"""
module CBandits

export ObjectiveFunc, DistributionFunc, SincFunc
export NarrowBump, WideBump, TwoBumps, ThreeBumps, Sinc
export DistributionActions, UniformActions, GaussianActions
export RandomBandit, RandomBanditResult, PWUCB, PWUCBResult, SBUCB, SBUCBResult
export GPUCBGrid, GPUCBGridResult, GPUCB, GPUCBResult
export metadata
export BanditSim, MetricStudy, MetricStudyResult, generate_sim_q, run_study, SweepStudy, GPkMetricStudy, GPkMetricStudyResult

using POMDPs, POMDPToolbox
using Distributions, StatsBase
using Plots; pyplot()
using Parameters
using ScikitLearnBase, Optim

include("objectives.jl")
include("action_dists.jl")
include("algorithms.jl")
include("sim.jl")
include("studies.jl")


end # module
