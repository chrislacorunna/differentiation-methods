import LinearAlgebra: diagm
diagonal(v::Vector{<:Real}) =
 diagm(0 => v)

import LinearAlgebra: I
eye(n::Integer) =
 Matrix(1.0I, n, n)

function ∇W(x, x̂, ŷ, y, Wo)
 ȳ = Wo * x̂
 # mean_squared_loss
 Eŷ = 2(ŷ - y)
 # liniowa funkcja aktywacji
 ŷȳ = ȳ |> length |> eye
 # sumowanie (W*x) wzg. wag
 ȳWo = x̂ |> transpose
 # sumowanie (W*x) wzg. wektora wej
 ȳx̂ = Wo |> transpose
 # sigmoidalna f. aktywacji
 x̂x̄ = x̂ .* (1.0 .- x̂) |> diagonal
 # sumowanie (W*x) wzg. wag
 x̄Wh = x |> transpose
 # reguła łańcuchowa
 Eȳ = ŷȳ * Eŷ
 Ex̂ = ȳx̂ * Eȳ
 Ex̄ = x̂x̄ * Ex̂
 EWo = Eȳ * ȳWo
 EWh = Ex̄ * x̄Wh
 return EWo, EWh
end

function net(x, wh, wo, y, dWo, dWh, Wo)
 x̂ = dense(wh, 3, 2, x, σ)
 ŷ = dense(wo, 1, 3, x̂, linear)
 @show ŷ

 EWo, EWh = ∇W(x, x̂, ŷ, y, Wo)
 dWo .= EWo
 dWh .= EWh

 E = mean_squared_loss(y, ŷ)
 println(E)
 println()
 #println(dWo)
 #println(dWh)
 #println(ŷ)
 #println(x̂)
end

dense(w, n, m, v, f) = f.(reshape(w, n, m) * v)
mean_squared_loss(y, ŷ) = sum((y - ŷ).^2)
σ(x) = one(x) / (one(x) + exp(-x))
linear(x) = x

function main()
  println("-----------------------------------------")
  @show Wh = [0.1965 -0.3744; -0.2722 1.6953; 1.1232 -0.898]
  @show Wo = [0.2102 -1.1416 -0.3463]
  println()
  x = [1.98;4.434]
  y = [0.064]
  dWo = similar(Wo)
  dWh = similar(Wh)

  epochs = 10
  for i=1:epochs
    println(dWh)
    println(dWo)
    println(Wh)
    net(x, Wh[:], Wo[:], y, dWo, dWh, Wo)
    Wh -= 0.1dWh
    Wo -= 0.1dWo
  end
end

main()
