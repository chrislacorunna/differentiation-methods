import Pkg
Pkg.activate(".")
MODULE_PATH = string(pwd(), "/backward/src")
if MODULE_PATH ∉ LOAD_PATH
    push!(LOAD_PATH, MODULE_PATH)
end

using AutomaticDiff, LinearAlgebra

dense(w, n, m, v, f) = f.(w * v)

println("-----------------------------------------")
@show x1 = Variable(copy(transpose([1. 1.])))
@show x2 = Variable(copy(transpose([2. 3.])))
@show Wo = Variable([1. 4.; 2. 5.; 3. 6.])

x̂ = mean_squared_loss.(x1, x2)
y = dense(Wo, 3, 2, x̂, linear)
@show y.output
backward(y)
@show x1.grad, x2.grad
