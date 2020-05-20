import Pkg
Pkg.activate(".")
MODULE_PATH = string(pwd(), "/backward/src")
if MODULE_PATH ∉ LOAD_PATH
    push!(LOAD_PATH, MODULE_PATH)
end

using AutomaticDiff, LinearAlgebra

dense(w, n, m, v, f) = f.(w * v)

println("-----------------------------------------")
@show x1 = Variable([log(1); log(2); log(3); log(4)])
y = softmax(x1)
@show y.output
backward(y)
@show x1.grad
