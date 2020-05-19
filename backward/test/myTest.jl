Pkg.activate(".")
MODULE_PATH = string(pwd(), "/backward/src")
if MODULE_PATH ∉ LOAD_PATH
    push!(LOAD_PATH, MODULE_PATH)
end

using AutomaticDiff, LinearAlgebra

function net(x, Wh, Wo, y)
  x̂ = dense(Wh, 3, 2, x, linear)
  #x̂ = linear.(x)
  @show x̂.output
  #x̂ = [0.21908276334329121, 0.999068706359944, 0.1470665901741307]
  ŷ = dense(Wo, 2, 3, x̂, linear)
  @show ŷ.output
  E = mean_squared_loss.(y, ŷ)
  @show E.output
  backward(E)
  @show Wh.grad, Wo.grad
end

dense(w, n, m, v, f) = f.(w * v)
σ(x) = one(x) / (one(x) + exp(-x))

function main()
  println("-----------------------------------------")
  @show Wh = Variable([1. 1.; 1. 1.; 1. 1.])
  @show Wo = Variable([1. 2. 3.; 2. 4. 6.])
  println()
  @show x = Variable(copy(transpose([1. 1.])))
  @show y = copy(transpose([0. 0.]))
  @show typeof(x)
  @show typeof(y)
  dWo = similar(Wo)
  dWh = similar(Wh)

  epochs = 1
  for i=1:epochs
    #Wh2 = Variable(Wh.value[:]) # TODO: clean it up
    #Wo2 = Variable(Wo.value[:])
    net(x, Wh, Wo, y)
    #Wh -= 0.1dWh
    Wo -= 0.1dWo
  end
end

main()
