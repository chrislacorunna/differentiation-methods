module ForwardDiff
export forward

using Global

import DualNum.Dual, DualNum.partials
J = function jacobian(fun, layers::Array{Layer, 1}, ind, benchmark)
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
        column = partials.([fun(layers, benchmark)...])
        push!(jacobian_columns, column[:,:])
    end
    return hcat(jacobian_columns...)
end

end #module ForwardDiff
