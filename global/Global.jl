#TODO: rename
module Global
export dense, mean_squared_loss, linear, σ, ReLU, softmax, Layer

dense(w, v, f) = (f == softmax ? f(w * v) : f.(w * v))

mean_squared_loss(y, ŷ) = sum((y - ŷ).^2)
linear(x) = x
σ(x) = one(x) / (one(x) + exp(-x))
ReLU(x) = x > zero(x) ? x : zero(x)
function softmax(x)
    res = similar(x)
    weight_sum = 0.
    for i in 1:size(x, 1)
        for j in 1:size(x, 2)
            weight_val = ℯ^(x[i, j])
            res[i, j] = weight_val
            weight_sum += weight_val
        end
    end
    res = res ./ weight_sum
end

import DualNum.Dual
mutable struct Layer{FT}
    W::Union{Array{Float64, 2}, Array{Dual{Float64}, 2}}
    X::Union{Nothing, Array{Float64, 2}}
    f::FT

    Layer(W, f) = new{typeof(f)}(W, nothing, f)
    Layer(W, X, f) = new{typeof(f)}(W, X, f)
end

end  # module Global
