export RegressionModel, LinearRegressionModel, LogisticRegressionModel

using ForwardDiff, LinearAlgebra

mutable struct RegressionModel <: AbstractMultObjModel
  meta :: MultObjNLPMeta
  counters :: MultObjCounters

  X :: Matrix
  y :: Vector
  h    # h(x, w)
  ℓ    # ℓ(y, ŷ)
end

function RegressionModel(X, y, h, ℓ)
  n, p = size(X)

  meta = MultObjNLPMeta(p, n)
  counters = MultObjCounters(n)
  return RegressionModel(meta, counters, X, y, h, ℓ)
end

function LinearRegressionModel(X, y)
  h(x, β) = dot(x,β)
  ℓ(y, ŷ) = (y - ŷ)^2 / 2
  return RegressionModel(X, y, h, ℓ)
end

function LogisticRegressionModel(X, y)
  h(x, β) = 1 / (1 + exp(-dot(x, β)))
  ℓ(y, ŷ) = -y * log(ŷ + 1e-8) - (1 - y) * log(1 - ŷ + 1e-8)
  return RegressionModel(X, y, h, ℓ)
end

function MultObjNLPModels.obj(nlp :: RegressionModel, i :: Integer, β :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  increment!(nlp, :neval_obji, i)
  return nlp.ℓ(nlp.y[i], nlp.h(nlp.X[i,:], β))
end

function MultObjNLPModels.grad!(nlp :: RegressionModel, i :: Integer, β :: AbstractVector , g :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β g
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  increment!(nlp, :neval_gradi, i)
  ForwardDiff.gradient!(g, β->nlp.ℓ(nlp.y[i], nlp.h(nlp.X[i,:], β)), β)
  return g
end