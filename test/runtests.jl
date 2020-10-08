using LinearAlgebra
using MultObjNLPModels
using Test

include("manual_rosenbr.jl")

@testset "Rosenbrock" begin
  nlp = Rosenbrock()

  x = ones(2)
  @test obj(nlp, x) ≈ 0
  @test norm(grad(nlp, x)) ≈ 0
end