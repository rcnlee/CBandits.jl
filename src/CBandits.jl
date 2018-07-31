"""
A package for bandit algorithms in 1D continuous space
"""
module CBandits

export ObjectiveFunc, Objective1D, Objective2D, DistributionFunc, SincFunc
export NarrowBump, WideBump, TwoBumps, ThreeBumps, Sinc
export DistributionFunc2D, NarrowBump2D, WideBump2D, Rosenbrock
export DistributionActions, UniformActions, GaussianActions, GridActions
export DistributionActions2D, UniformActions2D, GaussianActions2D, GridActions2D
export RandomBandit, RandomBanditResult, PWUCB, PWUCBResult, SBUCB, SBUCBResult
export GPUCBGrid, GPUCBGridResult, GPUCB, GPUCBResult
export metadata
export BanditSim, MetricStudy, MetricStudyResult, generate_sim_q, run_study, SweepStudy 
export GPkMetricStudy, GPkMetricStudyResult, GPOptimMetricStudy, GPOptimMetricStudyResult

using POMDPs, POMDPToolbox
using Distributions, StatsBase
using Plots; pyplot()
using Parameters
using GaussianProcesses
using ScikitLearnBase, Optim
using TestFunctions

include("objectives.jl")
include("action_dists.jl")
include("algorithms.jl")
include("sim.jl")
include("studies.jl")


end # module
