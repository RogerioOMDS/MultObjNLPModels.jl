export AbstractMultObjModel,
       obj, grad!, hess_structure!, hess_coord!, hprod!,
       increment!

"""
    AbstractMultObjModel

`AbstractMultObjModel` is an abstract type for problems with multiple objective.
The default way of looking into these models is by defining the objective

```math
f(x) = ∑ᵢ σᵢ fᵢ(x)
````

An `AbstractMultObjModel` is expected to have:
- `meta :: MultObjNLPMeta`
"""
abstract type AbstractMultObjModel <: AbstractNLPModel end

sum_counters(nls :: AbstractMultObjModel) = sum_counters(nls.counters)

import Base.show
show_header(io :: IO, nlp :: AbstractMultObjModel) = println(io, typeof(nlp))

function show(io :: IO, nlp :: AbstractMultObjModel)
  show_header(io, nlp)
  show(io, nlp.meta)
  show(io, nlp.counters)
end

for counter in fieldnames(MultObjCounters)
  counter == :counters && continue
  @eval begin
    """
    $($counter)(nlp)

    Get the number of `$(split("$($counter)", "_")[2])` evaluations.
    """
    $counter(nls :: AbstractMultObjModel) = nls.counters.$counter
    export $counter
  end
end

for counter in fieldnames(Counters)
  @eval begin
    $counter(nls :: AbstractMultObjModel) = nls.counters.counters.$counter
    export $counter
  end
end

"""
    increment!(nlp, s, i)

Increment counter `s[i]` of problem `nlp`.
"""
function increment!(nlp :: AbstractMultObjModel, s :: Symbol, i :: Integer)
  getproperty(nlp.counters, s)[i] += 1
end

function NLPModels.reset!(nls :: AbstractMultObjModel)
  reset!(nls.counters)
  return nls
end

# New multiple objective functions:

"""
    f = obj(nlp, i, x)

Evaluate ``fᵢ(x)``, the i-th objective function of `nlp` at `x`.
"""
function obj end

"""
    g = grad!(nlp, i, x, g)

Evaluate ``∇fᵢ(x)``, the gradient of the i-th objective function at `x` in place.
"""
function grad! end

"""
    hess_structure!(nlp, i, rows, cols)

Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
"""
function hess_structure! end

"""
    vals = hess_coord!(nlp, i, x, vals)

Evaluate the i-th objective Hessian at `x` in sparse coordinate format,
i.e., ``∇²fᵢ(x)``, rewriting `vals`.
Only the lower triangle is returned.
"""
function hess_coord! end

"""
    Hv = hprod!(nlp, i, x, v, Hv)

Evaluate the product of the i-th objective Hessian at `x` with the vector `v` in
place, where the objective Hessian is ``∇²fᵢ(x)``.
"""
function hprod! end

# NLPModels functions

function NLPModels.obj(nlp :: AbstractMultObjModel, x :: AbstractVector)
  return sum(obj(nlp, i, x) * nlp.meta.weights[i] for i = 1:nlp.meta.nobj)
end

function NLPModels.grad!(nlp :: AbstractMultObjModel, x :: AbstractVector{T}, g :: AbstractVector) where T
  fill!(g, zero(T))
  gi = fill!(similar(g), zero(T))
  for i = 1:nlp.meta.nobj
    grad!(nlp, i, x, gi)
    g .+= nlp.meta.weights[i] * gi
  end
  return g
end

function NLPModels.hess_structure!(nlp :: AbstractMultObjModel, hrows :: AbstractVector{<: Integer}, hcols :: AbstractVector{<: Integer})
end

function NLPModels.hess_coord!(nlp :: AbstractMultObjModel, x :: AbstractVector{T}, hvals :: AbstractVector; obj_weight :: Real=one(T)) where T
end

function NLPModels.hess_coord!(nlp :: AbstractMultObjModel, x :: AbstractVector{T}, y :: AbstractVector, hvals :: AbstractVector; obj_weight :: Real=one(T)) where T
end

function NLPModels.hprod!(nlp :: AbstractMultObjModel, x :: AbstractVector{T}, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real=one(T)) where T
  fill!(Hv, zero(T))
  Hiv = fill!(similar(Hv), zero(T))
  for i = 1:nlp.meta.nobj
    hprod!(nlp, i, x, v, Hiv)
    Hv .+= nlp.meta.weights[i] * Hiv
  end
  return Hv
end

function NLPModels.hprod!(nlp :: AbstractMultObjModel, x :: AbstractVector{T}, y :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real=one(T)) where T
end