export RegressionModel, LinearRegressionModel, LogisticRegressionModel

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
  h(x, β) = 1 / (1 + exp(-dot(x,β)))
  ℓ(y, ŷ) = -y * log(ŷ) - (1 - y) * log(1 - ŷ)
  return RegressionModel(X, y, h, ℓ)
end

  