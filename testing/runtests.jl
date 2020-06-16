import Pkg
Pkg.activate(".")

for PATH in ("/global", "/testing", "/manual", "/forward", "/backward/test", "/backward/src")
    MODULE_PATH = string(pwd(), PATH)
    if MODULE_PATH ∉ LOAD_PATH  push!(LOAD_PATH, MODULE_PATH) end
end

using Global, TestTools, ForwardTest, BackwardTest

test_cases = TestCase[]
test_data = Layer[]
#use arr2d to define matrices
Wh = arr2d([0.1965 -0.3744; -0.2722 1.6953; 1.1232 -0.898])
x = arr2d([1.98;4.434])
# pass inputs to the first layer
push!(test_data, Layer(Wh, x, σ))

Wo = arr2d([0.2102 -1.1416 -0.3463])
# pass only the coefficients and the activation function to further layers
push!(test_data, Layer(Wo, linear))

y = arr2d([0.064])
#= if using mean_squared_loss, use it in the last layer;
    pass targer vector and mean_squared_loss to the last layer =#
push!(test_data, Layer(y, mean_squared_loss))

#= test_funs is a list of references to the actual testing functions for respective
    differentiation methods (forward/backward);

    neuralnet_test functions are generally for the benchmarking purposes;
    hovewer, the same functions can be used to test whether a function gives correct results -
        in that case use a 1-layer network with an identity matrix as W or X =#
test_funs = TestFun[]
push!(test_funs, TestFun("Forward", ForwardTest.neuralnet_test))
push!(test_funs, TestFun("Backward", BackwardTest.neuralnet_test))

push!(test_cases, (TestCase(() -> neuralnet_test(test_data, test_funs))))

run_tests(test_cases)

#= the results of this one specific network can be also compared with the manual version
    (proof of general correctness) =#

#using ManualDiff
#println("-----------------------------------------")
#println("Manual")
#println("-----------------------------------------")
#manual()
