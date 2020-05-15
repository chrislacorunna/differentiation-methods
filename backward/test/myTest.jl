Pkg.activate(".")
MODULE_PATH = string(pwd(), "/src")
if MODULE_PATH ∉ LOAD_PATH
    push!(LOAD_PATH, MODULE_PATH)
end

using AutomaticDiff, LinearAlgebra

function net(x, Wh, Wo, y)
  x̂ = dense(Wh, 3, 2, x, linear)
  @show x̂.output
  #x̂ = [0.21908276334329121, 0.999068706359944, 0.1470665901741307]
  ŷ = dense(Wo, 1, 3, x̂, linear)
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
  @show Wh = Variable([0.1965 -0.3744; -0.2722 1.6953; 1.1232 -0.898])
  @show Wo = Variable([0.2102 -1.1416 -0.3463])
  println()
  @show x = copy(transpose([1.98 4.434]))
  @show y = copy(transpose([0.064]))
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
