export grad_desc, grad_estoc, MultObjNLPModels.obj, MultObjNLPModels.grad!


function grad_desc(nlp::RegressaoModel; 
    γ = 1e-2,
    max_eval=100000,
    max_time = 60.0,
    atol = 1e-10,
    rtol = 1e-10 
    )
    
    start_time = time()
    iter = 0
    β = nlp.meta.x0
    ∇ℓ(x,y,β) = ForwardDiff.gradient(β->nlp.ℓ(y, nlp.h(x,β)), β) #rever
    f = nlp.ℓ(nlp.y, nlp.h(nlp.X,β))
    N = norm(nlp.ℓ(nlp.y, nlp.h(nlp.X,β)))
  
    ϵt = atol + rtol*N
    Δt = time() - start_time   
    solved = N < ϵt 
    tired = (Δt > max_time) || (iter > max_eval)
  
    @info log_header([:iter, :f],
    [Int, Float64],
    hdr_override=Dict(:f => "f(β)"))
    
    @info log_row(Any[iter, f])
    
    while !(solved || tired)
      β -= γ * ∇ℓ(X,y,β)
  
      Δt = time() - start_time
      N = norm(nlp.ℓ(nlp.y, nlp.h(nlp.X,β)))
      solved = N < ϵt
      tired = (Δt > max_time) || (iter > max_eval) 
      iter+=1
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
  
    # return GenericExecutionStats(status, nlp; 
    #                               solution = β, 
    #                               objective=f, 
    #                               elapsed_time=Δt,
    #                               iter=iter)
    return X*β
  end
  

function grad_estoc(nlp::RegressaoModel;
                      γ = 1e-2,
                      max_eval=100000,
                      max_time = 60.0,
                      atol = 1e-10,
                      rtol = 1e-10 
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
      β -= γ * MultObjNLPModels.grad!(nlp, i, β)
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
  
    # return GenericExecutionStats(status, nlp; 
    #                               solution = β, 
    #                               objective=f, 
    #                               elapsed_time=Δt,
    #                               iter=iter)

    y_pred = [nlp.h(nlp.X[i,:],β) > 0.5 ? 1.0 : 0.0 for i=1:n ]
    return y_pred
  end
  
  
  
  function MultObjNLPModels.obj(nlp :: RegressaoModel, i :: Integer, β :: AbstractVector)
    # @lencheck alguma coisa
    return nlp.ℓ(nlp.y[i], h(nlp.X[i,:],β )) # ℓ( yᵢ, h(xᵢᵀβ) ) 
  end
  
  function MultObjNLPModels.grad!(nlp :: RegressaoModel, i :: Integer, x :: AbstractVector #= , g :: AbstractVector =#)
    # NLPModels.@lencheck 2 x g
    # NLPModels.@rangecheck 1 2 i
    # increment!(nlp, :neval_gradi, i)
    ∇f(x,y,β) = ForwardDiff.gradient(β->nlp.ℓ(y, nlp.h(x,β)), β)
    return ∇f(nlp.X[i,:], nlp.y[i], x)
  end
  