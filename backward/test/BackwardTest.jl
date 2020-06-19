module BackwardTest
using Statistics, Global, TestTools, BackwardDiff

function neuralnet_test(layers::Array{Layer, 1}, epochs, benchmark)
    grads = Array{Float64, 2}[]
    grads_raw = Array{Float64, 2}[]
    variables = Variable[]
    for l in layers
        if typeof(l) != Layer{typeof(mean_squared_loss)}
            push!(variables, Variable(l.W))
            push!(grads, similar(l.W))
            push!(grads_raw, Array{Float64}(undef, 0, 0))
        end
    end

    x = undef
    for n = 1:epochs
        # assume that mean_squared_loss can be only used at the last layer
        x = undef
        first = true
        for i = 1:size(layers, 1)
            l = layers[i]
            if first    x = l.X; first = false end
            typeof(l) == Layer{typeof(mean_squared_loss)} ?
                x = mean_squared_loss.(l.W, x) :
                x = dense(variables[i], x, l.f)
        end
        backward(x)
        if !benchmark   println("Result after epoch $n:\t$(x.output)") end
        for i = 1:size(grads, 1)
            grads_raw[i] = grad(variables[i])
            grads[i][:] = mean(grads_raw[i], dims=1)
        end
        # assume that mean_squared_loss can be only used at the last layer
        for i = 1:size(grads, 1)
            variables[i].value -= 0.1grads[i]
        end
    end
    if !benchmark
        println("\nCoefficients after error minimization:")
        for i = 1:size(grads, 1)
            println("Layer $i:\t$(value(variables[i]))")
        end
    end
    grads_raw
end

end # module
