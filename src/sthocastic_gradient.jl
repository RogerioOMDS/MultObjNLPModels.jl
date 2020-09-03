export sthocastic_gradient , MultObjNLPModels.obj, MultObjNLPModels.grad!


function sthocastic_gradient(
   nlp::RegressionModel;
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

  # f = nlp.ℓ(y, nlp.h(X,β))
  # N = norm(nlp.ℓ(y, nlp.h(X,β)))
  
  ϵt = atol + rtol*N
  Δt = time() - start_time   
  solved = N < ϵt 
  tired = (Δt > max_time) || (iter > max_eval)
  
  @info log_header([:iter, :f],
                   [Int, Float64],
                   hdr_override=Dict(:f => "f(β)"))
    
  @info log_row(Any[iter, f])
    
  # while !(solved || tired)
  for k = 1:10n
    for i in shuffle(1:n)
      β -= γ * MultObjNLPModels.grad!(nlp, i, β, g) #preciso arrumar esse g
    end
  end

  status = if solved
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
  
    # return GenericExecutionStats(status, nlp;  #ainda não arrumei esse trecho porque não ta rodando
    #                               solution = β, 
    #                               objective=f, 
    #                               elapsed_time=Δt,
    #                               iter=iter)

    # y_pred = [nlp.h(nlp.X[i,:],β) > 0.5 ? 1.0 : 0.0 for i=1:n ]  
    return y_pred
end
  
  
  
function MultObjNLPModels.obj(nlp :: RegressaoModel, i :: Integer, β :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar x
  #NLPModels.@rangecheck 1 nlp.meta.nvar i #falta saber o que é o "1"
  increment!(nlp, :neval_obji, i)
  return nlp.ℓ(nlp.y[i], h(nlp.X[i,:],β )) # ℓ( yᵢ, h(xᵢᵀβ) ) 
end
  
 function MultObjNLPModels.grad!(nlp :: RegressaoModel, i :: Integer, x :: AbstractVector , g :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar x g
  #NLPModels.@rangecheck 1 nlp.meta.nvar i #falta saber o que é o "1"
  increment!(nlp, :neval_gradi, i)
  ∇f(x,y,β) = ForwardDiff.gradient!(g, β->nlp.ℓ(y, nlp.h(x,β)), β) #preciso arrumar esse g ainda
  #return ∇f(nlp.X[i,:], nlp.y[i], x)
  return g
end
