using LinearAlgebra, Random, SolverTools

export sthocastic_gradient

function sthocastic_gradient(
    nlp::AbstractMultObjModel;
    learning_rate::Symbol = :optimal,
    γ::Float64 = 1e-2,
    α::Float64 = 1e-2, # for penalty
    ρ::Float64 = 0.85, # just for elasticnet
    penalty::Symbol = :l2,
    max_eval::Int = 0,
    max_time::Float64 = 60.0,
    max_iter::Int = 100,
    atol::Float64 = 1e-8,
    rtol::Float64 = 1e-8,
    batch_size::Int = 1,
    ν::Float64 = 0.5,
    d::Float64 = 0.5, # for some learning rate,
    r::Int64 = 10, #step to decline
    γ0::Float64 = 1.0 # first step
  )

  iter = 0
  start_time = time()
  β = nlp.meta.x0
  n = nlp.meta.nobj
  p = nlp.meta.nvar
  βavg = similar(β)
  g = similar(β)

  f = NLPModels.obj(nlp, β)
  NLPModels.grad!(nlp, β, g)

  Δt = time() - start_time

  P = if penalty == :l2
    β -> β
  elseif penalty == :l1
    β -> sign.(β)
  elseif penalty == :elasticnet
    β -> ρ * β + (1 - ρ) * sign.(β)
  end

  if learning_rate == :optimal && α == 0.0
    @warn("Can't use learning_rate optimal with α = 0.0. Changing to learning_rate step_based.")
    learning_rate = :step_based
  end

  γ = if learning_rate == :optimal
    iter -> γ0 / (α * (1e3 + iter))
  elseif learning_rate == :invscaling
    iter -> γ0 / (iter + 1) ^ ν
  elseif learning_rate == :time_based
    iter -> γ0 / (1 + d * iter)
  elseif learning_rate == :step_based
    iter -> γ0 * d ^ div(1 + iter, r)
  elseif  learning_rate == :exponential
    iter -> γ0 / exp(d * iter)
  end

  status = :unknown
  tired = Δt > max_time || sum_counters(nlp) > max_eval > 0 || iter > max_iter
  small_step = γ(iter) < 1e-6

  βavg .= 0
  num_batches = ceil(Int, n / batch_size)

  # betas =[]
  # gamas = []
  while !(small_step || tired)

    gnorm = -1.0
    Rindex = shuffle(1:n)
    for l = 1:num_batches
      index = if l < num_batches
        batch_size * (l - 1) .+ (1:batch_size)
      else
        batch_size * (l - 1) + 1:n
      end

      grad!(nlp, Rindex[index], β, g)
      β -= γ(iter) * (α * P(β) + g / length(index))
      βavg += β
    end

    # append!(betas, β)
    # append!(gamas,γ(iter))

    βavg = βavg / num_batches
    Δt = time() - start_time
    iter += 1
    tired = Δt > max_time || sum_counters(nlp) > max_eval > 0 || iter > max_iter
    small_step = γ(iter) < 1e-6
  end

  status = if small_step
    :small_step
  elseif tired
    if Δt > max_time
      :max_time
    elseif sum_counters(nlp) > max_eval > 0
      :max_eval
    elseif iter > max_iter
      :max_iter
    end
  end

  # return iter, β, betas, gamas

  return GenericExecutionStats(status, nlp;
                               solution=β,
                               solver_specific=Dict(:βavg => βavg),
                               elapsed_time=Δt,
                               iter=iter
                               )
end