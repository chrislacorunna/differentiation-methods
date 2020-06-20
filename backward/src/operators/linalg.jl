using LinearAlgebra

function gradient(::typeof(*), grad, out_size, output, lhs::AbstractVecOrMat, rhs::AbstractVecOrMat)
    l1, l2 = size(lhs, 1), size(lhs, 2)
    r1, r2 = size(rhs, 1), size(rhs, 2)
    grad_l = zeros(out_size, l1 * l2)
    grad_r = zeros(out_size, r1 * r2)
    for i = 1:r1
        for j = 1:l1
            for k = 1:out_size
                grad_l[k, (i-1)*l1 + j] += rhs[i, 1] * grad[k, j]
                grad_r[k, i] += lhs[j, i] * grad[k, j]
            end
        end
    end
    (grad_l, grad_r)
end
