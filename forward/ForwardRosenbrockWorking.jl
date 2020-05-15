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

using Plots
function main()
  println("-----------------------------------------")
  rosenbrock(x, y) = (1.0 - x*x) + 100.0*(y - x*x)*(y - x*x)
  v = -1:.2:+1
  n = length(v)
  @show xv = repeat(v, inner=n)
  yv = repeat(v, outer=n)
  z = rosenbrock.(Dual.(xv, one.(xv)), yv)
  dx = 5e-4partials.(z)
  @show z = rosenbrock.(xv, Dual.(yv, one.(yv)))
  dy = 5e-4partials.(z)
  zv = value.(z)

  zv = reshape(zv, n, n)
  contour(v, v, zv, fill=true)
  quiver!(xv[:], yv[:], gradient=(dx, dy))
end

main()
