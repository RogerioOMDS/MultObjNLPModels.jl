export RegressionModel, LinearRegressionModel, LogisticRegressionModel

using ForwardDiff

mutable struct RegressionModel <: AbstractMultObjModel
  meta :: MultObjNLPMeta
  counters :: MultObjCounters

  X :: Matrix
  y :: Vector
  h    # h(x, w)
  ℓ    # ℓ(y, ŷ)
end

# function RegressionModel(X, y, f)
function RegressionModel(X, y, h, ℓ)
  n, p = size(X)

  meta = MultObjNLPMeta(p, n)
  counters = MultObjCounters(n)
  # return RegressionModel(meta, counters, X, y, f)
  return RegressionModel(meta, counters, X, y, h, ℓ)
end

function LinearRegressionModel(X, y)
  h(x, β) = dot(x, β)
  ℓ(y, ŷ) = (y - ŷ)^2 / 2
  # f(x, y, β) = ℓ(y, h(x, β))
  # return RegressionModel(X, y, f)
  return RegressionModel(X, y, h, ℓ)
end

function LogisticRegressionModel(X, y)
  h(x, β) = 1 / (1 + exp(-dot(x, β)))
  ℓ(y, ŷ) = -y * log(ŷ) - (1 - y) * log(1 - ŷ)
  # f(x, y, β) = ℓ(y, h(x, β))
  # return RegressionModel(X, y, f)
  return RegressionModel(X, y, h, ℓ)
end

# f(x) = ∑ᵢ σᵢ fᵢ(x)

function MultObjNLPModels.obj(nlp :: RegressionModel, i :: Integer, β :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  increment!(nlp, :neval_obji, i)
  return nlp.ℓ(nlp.y[i], nlp.h(nlp.X[i,:], β)) # ℓ( yᵢ, h(xᵢ, β) )
  # return nlp.f(X[i,:],y[i],β)
  # return nlp.obj(nlp.X) 
  # return nlp.f[i](β)
  # return nlp.f(β)[i]
end

 function MultObjNLPModels.grad!(nlp :: RegressionModel, i :: Integer, β :: AbstractVector , g :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β g
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  increment!(nlp, :neval_gradi, i)
  ForwardDiff.gradient!(g, β->nlp.ℓ(nlp.y[i], nlp.h(nlp.X[i,:], β)), β)
  # ForwardDiff.gradient!(g, nlp.f(X[i,:]),β)
  # ForwardDiff.gradient!(g, nlp.f[i], β)
  # ForwardDiff.gradient(nlp.f,β)[i]
  
  return g
end