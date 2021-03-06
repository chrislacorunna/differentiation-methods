module ManualDiff
export manual

using Global
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
    x̂ = dense(wh, x, σ)
    ŷ = dense(wo, x̂, linear)

    EWo, EWh = ∇W(x, x̂, ŷ, y, Wo)
    dWh .= EWh
    dWo .= EWo

    @show E = mean_squared_loss(y, ŷ)
end

function manual()
    Wh = [0.1965 -0.3744; -0.2722 1.6953; 1.1232 -0.898]
    Wo = [0.2102 -1.1416 -0.3463]
    x = [1.98;4.434]
    y = [0.064]
    dWo = similar(Wo)
    dWh = similar(Wh)

    epochs = 10
    for i=1:epochs
        net(x, Wh, Wo, y, dWo, dWh, Wo)
        Wh -= 0.1dWh
        Wo -= 0.1dWo
    end
    println("\nJacobian matrices for all coefficients after the last epoch:")
    println("Layer1:\t$Wh")
    println("Layer1:\t$Wo")
    println("\nCoefficients after the last epoch:")
    dWh = reshape(dWh, 1, length(dWh))
    dWo = reshape(dWo, 1, length(dWo))
    println("Layer1:\t$dWh")
    println("Layer1:\t$dWo")
end

end #module ManualDiff
