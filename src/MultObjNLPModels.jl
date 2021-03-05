module MultObjNLPModels

# stdlib
using SparseArrays

# external packages
using FastClosures

# JSO packages
using LinearOperators, NLPModels

include("api.jl")       # Defines the model and the functions
include("meta.jl")      # Defines the derived meta structure
include("counters.jl")

include("auxiliary.jl")

include("regression_model.jl")
include("sthocastic_gradient.jl")

end # module
