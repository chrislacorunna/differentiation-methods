module BackwardDiff

using DiffRules, Global

abstract type AbstractNode end
abstract type Operator end

# core
include("comput_graph.jl")

# operators
include("operators/broadcast.jl")
include("operators/linalg.jl")
include("operators/custom.jl")
include("operators/math.jl")

end # module BackwardDiff
