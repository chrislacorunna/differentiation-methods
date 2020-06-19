module ForwardTest

using Statistics, Global, TestTools

coeff(layer::Layer) = layer.W

import ForwardDiff.J
curr_epoch = 1
function neuralnet_test(layers::Array{Layer, 1}, epochs, benchmark)
    layers = deepcopy(layers)
    grads = Array{Float64, 2}[]
    grads_raw = Array{Float64, 2}[]
    for l in layers
        if typeof(l) != Layer{typeof(mean_squared_loss)}
            push!(grads, similar(l.W))
            push!(grads_raw, Array{Float64}(undef, 0, 0))
        end
    end

    for n = 1:epochs
        global curr_epoch = n
        for i = 1:size(layers, 1)
            if typeof(layers[i]) != Layer{typeof(mean_squared_loss)}
                layers_copy = deepcopy(layers)
                grads_raw[i] = J(net, layers_copy, i, benchmark)
                grads[i][:] = mean(grads_raw[i], dims=1)
            end
        end
        global first_call = true

        # assume that mean_squared_loss can be only used at the last layer
        for i = 1:size(grads, 1)
            layers[i].W -= 0.1grads[i]
        end
    end
    if !benchmark
            println("\nCoefficients after error minimization:")
        for i = 1:size(grads, 1)
            println("Layer $i:\t$(coeff(layers[i]))")
        end
    end
    grads_raw
end

import DualNum.value
first_call = true
function net(layers::Array{Layer, 1}, benchmark)
    first = true
    x = undef
    for l in layers
        if first    x = l.X; first = false end
        typeof(l.f) == typeof(mean_squared_loss) ?
            x = mean_squared_loss.(l.W, x) :
            x = dense(l.W, x, l.f)
    end

    #TODO: eval passed parameter or other automatic way to run verbose/silent for tests/benchmarks
    if !benchmark
        if first_call
            println("Result after epoch $curr_epoch:\t$(value.(x))");
            global first_call = false
        end
    end
    x
end

end #module ForwardTest
