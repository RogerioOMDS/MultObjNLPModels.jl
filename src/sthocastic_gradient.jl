using LinearAlgebra, Random, SolverTools

NLPModels.has_bounds(meta::MultObjNLPMeta) = length(meta.ifree) < meta.nvar 
NLPModels.unconstrained(meta::MultObjNLPMeta) = meta.ncon == 0 && !has_bounds(meta)

export sthocastic_gradient

function sthocastic_gradient(
    nlp::AbstractMultObjModel;
    γ = 1e-2,
    max_eval=10000,
    max_time = 60.0,
    atol = 1e-8,
    rtol = 1e-8
  )

  start_time = time()
  iter = 0
  β = nlp.meta.x0
  n = nlp.meta.nobj
  g = similar(β)

  # f = nlp.ℓ(nlp.y[i], nlp.h(nlp.X[i,:],β))
  # N = norm(nlp.ℓ(nlp.y, nlp.h(nlp.X,β)))

  # ϵt = atol + rtol*N
  Δt = time() - start_time
  # solved = N < ϵt
  tired = (Δt > max_time) || (iter > max_eval)

  # @info log_header([:iter, :f],
  #                  [Int, Float64],
  #                  hdr_override=Dict(:f => "f(β)"))

  # @info log_row(Any[iter, f])

  for k = 1:30n # I got better results with this number
    for i in shuffle(1:n)
      β -= γ * grad!(nlp, i, β, g) 
    end

    Δt = time() - start_time
    iter+=1
    tired = (Δt > max_time) || (iter > max_eval)
  end

  status = if (iter == 30n)
    :first_order
  elseif tired
    if Δt >: max_time
      :max_time
    else
      :max_eval
    end
  else
    :unknown
  end

  return GenericExecutionStats(status, nlp;  
                                solution = β,
                                #= objective=f, =#
                                elapsed_time=Δt,
                                iter=iter)
end