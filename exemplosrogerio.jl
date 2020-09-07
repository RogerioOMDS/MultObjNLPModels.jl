using NLPModels, MultObjNLPModels, Random, LinearAlgebra

function grad_desc(nlp::AbstractNLPModel;
    γ = 1e-2,
    max_eval=100000,
    max_time = 60.0,
    atol = 1e-6,
    rtol = 1e-6
    )

    start_time = time()
    iter = 0
    β = nlp.meta.x0
    ∇ℓ(x,y,β) = ForwardDiff.gradient(β->nlp.ℓ(y, nlp.h(x,β)), β) #nlp normal 
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
      # for i=1:nlp.meta.nvar
      #   β[i] -= γ * MultObjNLPModels.grad!(nlp, i, β)
      # end
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


# REGRESSÃO LINEAR
# h(β, x) = dot(β, x)
# ℓ(y, ŷ) = sum((y - ŷ).^2) / 2
# modelo = RegressaoModel(X, y, h, ℓ)

# n,p = [10,2]

# X = rand(n,p)
# y = [sum(X[i,:]).+randn()*0.5 for i=1:n]

# println("y=$y")
# output = LinearRegressionModel(X,y)

# println()
# println(output)
# println()
# println(grad_desc(output))

####################################################################

# REGRESSÃO LOGISTICA

n = 50

xl = sort(rand(n) * 2 .- 1)
yl = [x + 0.2 + randn() * 0.25 > 0 ? 1 : 0 for x in xl]
xl = hcat(ones(n), xl)
println()
println("y",yl)

pred = LogisticRegressionModel(xl,yl)

println(pred)
println()
println(sthocastic_gradient(pred))
