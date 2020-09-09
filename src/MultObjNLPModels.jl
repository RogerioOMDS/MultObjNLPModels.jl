module MultObjNLPModels

using NLPModels

include("meta.jl")
include("counters.jl")
include("api.jl")

include("regression_model.jl")
include("sthocastic_gradient.jl")
end # module