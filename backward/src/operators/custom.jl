export linear, mean_squared_loss

linear(x) = x
gradient(::typeof(linear), grad, out_size, output, x) = (grad, )

mean_squared_loss(y, ŷ) = sum((y - ŷ).^2)
#=
function gradient(::typeof(mean_squared_loss), grad, out_size, output, y, ŷ)
    (grad * (2*y - 2*ŷ), grad * (2*ŷ - 2*y))
end
=#
function gradient(mt::Trait.Broadcasted{typeof(mean_squared_loss)}, grad, out_size, output, y, ŷ)
    (@.(grad * (2*y - 2*ŷ)), @.(grad * (2*ŷ - 2*y)), )
end
