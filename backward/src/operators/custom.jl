export linear, σ, ReLU, softmax, mean_squared_loss

linear(x) = x
gradient(::Trait.Broadcasted{typeof(linear)}, grad, out_size, output, x) = (grad, )

σ(x) = one(x) / (one(x) + exp(-x))
function gradient(::Trait.Broadcasted{typeof(σ)}, grad, out_size, output, x)
    t_grad = transpose(grad)
    transpose.((@.(t_grad * (1 - σ(x)) * σ(x)), ))
end

ReLU(x) = x > zero(x) ? x : zero(x)
function gradient(::Trait.Broadcasted{typeof(ReLU)}, grad, out_size, output, x)
    (ifelse.(x .> 0., grad, grad * 0), )
end

softmax(x::AbstractNode) = register(softmax, x)
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
function gradient(::typeof(softmax), grad, out_size, output, x)
    jacob = Array{Float64, 2}(undef, size(x, 1), size(x, 1))
    for i in 1:size(x, 1)
        for j in 1:size(x, 1)
            if i == j
                jacob[i, j] = output[i] * (1 - output[j])
            else
                jacob[i, j] = -output[i] * output[j]
            end
        end
    end
    (grad * jacob, )
end

mean_squared_loss(y, ŷ) = sum((y - ŷ).^2)
#=
function gradient(::typeof(mean_squared_loss), grad, out_size, output, y, ŷ)
    (grad * (2*y - 2*ŷ), grad * (2*ŷ - 2*y))
end
=#
function gradient(mt::Trait.Broadcasted{typeof(mean_squared_loss)}, grad, out_size, output, y, ŷ)
    t_grad = transpose(grad)
    transpose.((@.(t_grad * (2*y - 2*ŷ)), @.(t_grad * (2*ŷ - 2*y))))
end
