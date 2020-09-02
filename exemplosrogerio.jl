using NLPModels, MultObjNLPModels, Random

# REGRESSÃO LINEAR
# h(β, x) = dot(β, x)
# ℓ(y, ŷ) = sum((y - ŷ).^2) / 2
# modelo = RegressaoModel(X, y, h, ℓ)

n,p = [10,2]

X = rand(n,p)
y = [sum(X[i,:]).+randn()*0.5 for i=1:n]

println("y=$y")
output = MultObjNLPModels.RegressaoLinearModel(X,y)

println()
# println(output)
println()
println(grad_desc(output))

####################################################################

# REGRESSÃO LOGISTICA 

# n = 50
# β = ones(2)
# xl = sort(rand(n) * 2 .- 1)
# yl = [x + 0.2 + randn() * 0.25 > 0 ? 1 : 0 for x in xl]
# xl = hcat(ones(n), xl)
# println()
# println("y",yl)

# pred = MultObjNLPModels.RegressaoLogisticaModel(xl,yl)

# println(pred)
# println()
# println(grad_estoc(pred))

