module TestTools
export TestFun, TestCase, run_tests, neuralnet_test, Layer, arr2d

using BenchmarkTools, Global, Dates
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5
arr2d(x::Array{T}) where T <: Number = reshape(x, size(x, 1), size(x, 2))

struct TestFun
    name::String
    ref
end

struct TestCase
    name::String
    fun
end

#WARNING: this assumes that if mean_squared_loss is used, it's at the last layer
function neuralnet_test(test_data::Array{Layer, 1}, test_funs::Array{TestFun, 1}, epochs; benchmark=false)
    for fun in test_funs
        println("-----------------------------------------")
        println(fun.name)
        println("-----------------------------------------")
        if benchmark
            global f = () -> fun.ref(test_data, epochs, benchmark)
            display(@benchmark f())
            println()
        else
            jacobians = fun.ref(test_data, epochs, benchmark)
        end
        if !benchmark
            println("\nJacobian matrices for all coefficients after the last epoch:")
            for i = 1:size(jacobians, 1)
                println("Layer $i:\t$(jacobians[i])")
            end
        end
    end
end

function run_tests(test_cases::Array{TestCase, 1})
    n = 1
    for case in test_cases
        println("--------------------------------------------------------")
        println("$(case.name)")
        case.fun()
        n+=1
    end
end

end #module TestTools
