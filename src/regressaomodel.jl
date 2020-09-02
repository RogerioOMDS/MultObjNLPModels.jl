export RegressaoModel, RegressaoLinearModel, RegressaoLogisticaModel

mutable struct RegressaoModel <: AbstractMultObjModel
    meta :: MultObjNLPMeta
    counters :: MultObjCounters
  
    X :: Matrix
    y :: Vector
    h    # h(x, w)
    ℓ    # ℓ(y, ŷ)
  end

function RegressaoModel(X, y, h, ℓ)
    n, p = size(X)
    
    meta = MultObjNLPMeta(p, n)
    counters = MultObjCounters(n)
    return RegressaoModel(meta, counters, X, y, h, ℓ)
  end
  
  function RegressaoLinearModel(X, y)
    h(x, β) = x*β
    ℓ(y, ŷ) = sum((y - ŷ).^2) / 2
    return RegressaoModel(X, y, h, ℓ) 
  end
  
  function RegressaoLogisticaModel(X, y)
    h(x, β) = 1 / (1 + exp(-x'*β))
    ℓ(y, ŷ) = -y * log(ŷ) - (1 - y) * log(1 - ŷ)
    return RegressaoModel(X, y, h, ℓ)
  end

  