export softmax
import Global.softmax
using Global

gradient(::Trait.Broadcasted{typeof(linear)}, grad, out_size, output, x) = (grad, )

function gradient(::Trait.Broadcasted{typeof(σ)}, grad, out_size, output, x)
    t_grad = transpose(grad)
    transpose.((@.(t_grad * (1 - σ(x)) * σ(x)), ))
end

function gradient(::Trait.Broadcasted{typeof(ReLU)}, grad, out_size, output, x)
    (ifelse.(x .> 0., grad, grad * 0), )
end

softmax(x::AbstractNode) = register(softmax, x)
function gradient(::typeof(softmax), grad, out_size, output, x)
    jacob = Array{Float64, 2}(undef, size(x, 1), size(x, 1))
    for i in 1:size(x, 1)
        for j in 1:size(x, 1)
            if i == j
                jacob[i, j] = output[i] * (1 - output[i])
            else
                jacob[i, j] = -output[i] * output[j]
            end
        end
    end
    (grad * jacob, )
end

#=
function gradient(::typeof(mean_squared_loss), grad, out_size, output, y, ŷ)
    (grad * (2*y - 2*ŷ), grad * (2*ŷ - 2*y))
end
=#
function gradient(mt::Trait.Broadcasted{typeof(mean_squared_loss)}, grad, out_size, output, y, ŷ)
    t_grad = transpose(grad)
    transpose.((@.(t_grad * (2*y - 2*ŷ)), @.(t_grad * (2*ŷ - 2*y))))
end
