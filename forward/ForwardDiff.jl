module ForwardDiff
export forward

using Global

#TODO: store all partial results and use them
import DualNum.Dual, DualNum.partials
J = function jacobian(fun, layers::Array{Layer, 1}, ind)
    coeffs = layers[ind].W
    a = size(coeffs, 1)
    b = size(coeffs, 2)
    coeffs = coeffs[:]
    jacobian_columns = Matrix{Float64}[]
    for i=1:length(coeffs)
        w = Dual{Float64}[]
        for j=1:length(coeffs)
            seed = (i == j)
            push!(w, seed ?
                Dual(coeffs[j], one(coeffs[j])) :
                Dual(coeffs[j], zero(coeffs[j])) )
        end
        layers[ind].W = reshape(w, a, b)
        column = partials.([fun(layers)...])
        push!(jacobian_columns, column[:,:])
    end
    return hcat(jacobian_columns...)
end

using Statistics
function forward()
    println("-----------------------------------------")
    Wh = [0.1965 -0.3744; -0.2722 1.6953; 1.1232 -0.898]
    Wo = [0.2102 -1.1416 -0.3463]
    println()
    x = [1.98;4.434]
    y = [0.064]
    dWh = similar(Wh)
    dWo = similar(Wo)

    epochs = 1
    for i=1:epochs
        # TODO: try @views
        dWh[:] = mean(dnet_Wh(x, Wh, Wo, y), dims=1)
        #dWo[:] = mean(dnet_Wo(x, Wh, Wo, y), dims=1)
        @show dWh
        #@show dWo
        Wh -= 0.1dWh
        #Wo -= 0.1dWo
        global first = true
    end
    #@show Wh
    #@show Wo
end

end #module ForwardDiff
