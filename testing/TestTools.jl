module TestTools
export TestFun, TestCase, run_tests, neuralnet_test, Layer, arr2d

using Global

arr2d(x::Array{T}) where T <: Number = reshape(x, size(x, 1), size(x, 2))

struct TestFun
    name::String
    ref
end

#TODO: test_f[]::typeof(testing_fun)
struct TestCase
    fun
end

#WARNING: this assumes that if mean_squared_loss is used, it's at the last layer
function neuralnet_test(test_data::Array{Layer, 1}, test_funs::Array{TestFun, 1})
    epochs = 10
    for fun in test_funs
        println("-----------------------------------------")
        println(fun.name)
        println("-----------------------------------------")
        jacobians = fun.ref(test_data, epochs)
        println("\nJacobian matrices for all coefficients after error minimization:")
        for i = 1:size(jacobians, 1)
            println("Layer $i:\t$(jacobians[i])")
        end
    end
end

function run_tests(test_cases::Array{TestCase, 1})
    n = 1
    for case in test_cases
        println("-----------------------------------------")
        println("Test $n")
        case.fun()
        n+=1
    end
end

end #module TestTools
