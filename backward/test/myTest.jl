import Pkg
Pkg.activate(".")
MODULE_PATH = string(pwd(), "/backward/src")
if MODULE_PATH ∉ LOAD_PATH
    push!(LOAD_PATH, MODULE_PATH)
end

using AutomaticDiff, LinearAlgebra

function net(x, Wh, Wo, Wz, Wzz, y)
    x̂ = dense(Wh, 3, 2, x, σ)
    @show x̂.output
    ŷ = dense(Wo, 2, 3, x̂, linear)
    @show ŷ.output
    z = dense(Wz, 3, 2, ŷ, softmax)
    @show z.output
    zz = dense(Wzz, 4, 3, z, ReLU)
    @show zz.output
    E = mean_squared_loss.(y, zz)
    @show E.output
    backward(E)
    @show Wh.grad
    @show Wo.grad
    @show Wz.grad
    @show Wzz.grad
end

dense(w, n, m, v, f) = (f == softmax ? f(w * v) : f.(w * v))

function main()
    println("-----------------------------------------")
    @show Wh = Variable([1. 1.; 1. 1.; 1. 1.])
    @show Wo = Variable([1. 2. 3.; 2. 4. 6.])
    @show Wz = Variable([3. 5.; -3. -6; 4. -1.])
    @show Wzz = Variable([3. -2. 4.; 2. 7. 1.; 4. -4. 2.; 6. 5. 3.])
    println()
    @show x = Variable([1.; 1.])
    @show y = copy(transpose([0. 0. 1. 1.]))
    @show typeof(y)
    dWo = similar(Wo)
    dWh = similar(Wh)

    epochs = 1
    for i=1:epochs
        net(x, Wh, Wo, Wz, Wzz, y)
        #dWh = ...
        #dWo = ...
        #Wh -= 0.1dWh
        #Wo -= 0.1dWo
    end
end

main()
