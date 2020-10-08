using LinearAlgebra, Random, SolverTools

NLPModels.has_bounds(meta::MultObjNLPMeta) = length(meta.ifree) < meta.nvar 
NLPModels.unconstrained(meta::MultObjNLPMeta) = meta.ncon == 0 && !has_bounds(meta)

export sthocastic_gradient

function sthocastic_gradient(
    nlp::AbstractMultObjModel;
    learning_rate::Symbol = :optimal, 
    γ::Float64 = 1e-2, 
    α::Float64 = 1e-2, # for penalty
    ρ::Float64 = 0.85, # just for elasticnet
    penalty::Symbol = :l2, 
    max_eval::Int = 50,
    max_time::Float64 = 60.0,
    atol::Float64 = 1e-8,
    rtol::Float64 = 1e-8,
    power_t::Float64 = 1e-2
  )

  iter = 0
  start_time = time()
  β = nlp.meta.x0
  βavg = similar(β)
  n = nlp.meta.nobj
  g = similar(β)
  f = NLPModels.obj(nlp, β)
  NLPModels.grad!(nlp,β,g)
  
  Δt = time() - start_time
  
  tired = (Δt > max_time) 
  itermax = iter > max_eval
  
  @info log_header([:iter, :f],
                   [Int, Float64],
                   hdr_override=Dict(:f => "f(β)"))

  @info log_row(Any[iter, f])

  
  P = if penalty == :l2
    β -> β 
  elseif penalty == :l1
    β -> sign.(β)
  elseif penalty == :elasticnet
    β -> ρ * β + (1-ρ) * sign.(β)
  end
  

  for k = 1:max_eval
  # while !(γ < 0.01 || tired) # sounds good. Doesnt work

    if learning_rate == :optimal
      γ = 1/(α * (1e3 + iter)) 
    elseif learning_rate == :invscaling 
      γ = 1e-2 / (iter+1)^(power_t)
    elseif learning_rate == :constant
      γ = γ    
    end

    βavg .= 0 # test later
    for i in shuffle(1:n)
      β -= γ * (α * P(β) + grad!(nlp, i, β, g))
      βavg += β # test later
    end
    βavg = βavg/n # test later

    Δt = time() - start_time
    tired = (Δt > max_time) 
    solved = (γ < 1e-2)
    itermax = iter > max_eval
    iter+=1
  end

  status = if (iter == max_eval)    
    :first_order
  elseif tired
    if Δt >: max_time
      :max_time
    elseif itermax 
      :max_eval
    end
  else
    :unknown
  end

  return GenericExecutionStats(status, nlp;  
                                solution = β,
                                solver_specific=Dict(:βavg=>βavg),
                                elapsed_time=Δt,
                                iter=iter
                                )
end
