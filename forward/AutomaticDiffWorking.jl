struct Dual{T <:Number} <:Number
  v::T
  dv::T
end

import Base: +, -, *, /
 -(x::Dual) = Dual(-x.v, -x.dv)
 +(x::Dual, y::Dual) = Dual( x.v + y.v, x.dv + y.dv)
 -(x::Dual, y::Dual) = Dual( x.v - y.v, x.dv - y.dv)
 *(x::Dual, y::Dual) = Dual( x.v * y.v, x.dv * y.v + x.v * y.dv)
 /(x::Dual, y::Dual) = Dual( x.v / y.v, (x.dv * y.v - x.v * y.dv)/y.v^2)
import Base: abs, sin, cos, tan, exp, sqrt, isless
abs(x::Dual) = Dual(abs(x.v),sign(x.v)*x.dv)
sin(x::Dual) = Dual(sin(x.v), cos(x.v)*x.dv)
cos(x::Dual) = Dual(cos(x.v),-sin(x.v)*x.dv)
tan(x::Dual) = Dual(tan(x.v), one(x.v)*x.dv + tan(x.v)^2*x.dv)
exp(x::Dual) = Dual(exp(x.v), exp(x.v)*x.dv)
sqrt(x::Dual) = Dual(sqrt(x.v),.5/sqrt(x.v) * x.dv)
isless(x::Dual, y::Dual) = x.v < y.v;

import Base: show
show(io::IO, x::Dual) = print(io, "(", x.v, ") + [", x.dv, "ϵ]");
value(x::Dual) = x.v;
partials(x::Dual) = x.dv;

import Base: convert, promote_rule
convert(::Type{Dual{T}}, x::Dual) where T =
 Dual(convert(T, x.v), convert(T, x.dv))

convert(::Type{Dual{T}}, x::Number) where T =
 Dual(convert(T, x), zero(T))

promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R} =
 Dual{promote_type(T,R)}

D = derivative(f, x) = partials(f(Dual(x, one(x))))

###
# real shit
dense(w, n, m, v, f) = f.(reshape(w, n, m) * v)
mean_squared_loss(y, ŷ) = sum((y - ŷ).^2)
σ(x) = one(x) / (one(x) + exp(-x))
linear(x) = x

J = function jacobian(f, args::Vector{T}) where {T <:Number}
  jacobian_columns = Matrix{T}[]
  for i=1:length(args)
    x = Dual{T}[]
    for j=1:length(args)
      seed = (i == j)
      push!(x, seed ?
        Dual(args[j], one(args[j])) :
        Dual(args[j],zero(args[j])) )
    end
    column = partials.([f(x)...])
    push!(jacobian_columns, column[:,:])
  end
  return hcat(jacobian_columns...)
end

function net(x, wh, wo, y)
  @show x̂ = dense(wh, 3, 2, x, σ)
  @show ŷ = dense(wo, 1, 3, x̂, linear)
  @show E = mean_squared_loss(y, ŷ)
end

dnet_Wh(x, wh, wo, y) = J(w -> net(x, w, wo, y), wh);
dnet_Wo(x, wh, wo, y) = J(w -> net(x, wh, w, y), wo);

function main()
  println("-----------------------------------------")
  @show Wh = [0.1965 -0.3744; -0.2722 1.6953; 1.1232 -0.898]
  @show Wo = [0.2102 -1.1416 -0.3463]
  println()
  x = [1.98;4.434]
  y = [0.064]
  dWh = similar(Wh)
  dWo = similar(Wo)

  epochs = 1
  for i=1:epochs
    @show dWh[:] = dnet_Wh(x, Wh[:], Wo[:], y)
    @show dWo[:] = dnet_Wo(x, Wh[:], Wo[:], y)
    @show dWh
    @show dWo
    Wh -= 0.1dWh
    Wo -= 0.1dWo
  end
end

main()
