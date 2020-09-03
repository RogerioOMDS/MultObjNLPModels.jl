module MultObjNLPModels

export RegressionModel, LinearRegressionModel, LogisticRegressionModel # vi que isso é feito no NLPModels, mas ainda assim não consegui 

using NLPModels

include("meta.jl")
include("counters.jl")
include("api.jl")

include("sthocastic_gradient.jl")
include("regression_model.jl")
end # module
