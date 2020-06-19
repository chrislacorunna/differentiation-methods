import Pkg
Pkg.activate(".")

for PATH in ("/global", "/testing", "/manual", "/forward", "/backward/test", "/backward/src")
    MODULE_PATH = string(pwd(), PATH)
    if MODULE_PATH ∉ LOAD_PATH  push!(LOAD_PATH, MODULE_PATH) end
end

function get_rand_activation()
    return rand([linear, ReLU, σ, softmax])
end
#function for testing random values. should be a tab where 1st element is
# x vector and next ones are neurons and the last one is output
function run_benchmark(neuron_tab, epochs)
    test_cases = TestCase[]
    test_data = Layer[]
    Wh = arr2d(randn(neuron_tab[2], neuron_tab[1]))
    x = arr2d(randn(neuron_tab[1], 1))
    push!(test_data, Layer(Wh, x, σ))

    for i = 3:size(neuron_tab,2)
        Wn = arr2d(randn(neuron_tab[i], neuron_tab[i-1]))
        push!(test_data, Layer(Wn, get_rand_activation()))
    end

    test_funs = TestFun[]
    push!(test_funs, TestFun("Forward", ForwardTest.neuralnet_test))
    push!(test_funs, TestFun("Backward", BackwardTest.neuralnet_test))

    push!(test_cases, (TestCase(() -> neuralnet_test(test_data, test_funs, epochs, benchmark = true))))

    run_tests(test_cases)
end

using Global, TestTools, ForwardTest, BackwardTest
#testing resource-dependency for epochs
# println("-----------------------------------------")
# println("Benchmarking 2 - 3 - 2 neural net with 1 epoch")
# run_benchmark(arr2d([2 3 2]), 1)
# println("-----------------------------------------")
# println("Benchmarking 2 - 3 - 2 neural net with 2 epochs")
# run_benchmark(arr2d([2 3 2]), 2)
# println("-----------------------------------------")
# println("Benchmarking 2 - 3 - 2 neural net with 8 epochs")
# run_benchmark(arr2d([2 3 2]), 8)
# println("Benchmarking 2 - 3 - 2 neural net with 32 epochs")
# run_benchmark(arr2d([2 3 2]), 32)
# println("Benchmarking 2 - 3 - 2 neural net with 128 epochs")
# run_benchmark(arr2d([2 3 2]), 128)

#testing layer-number dependency
# println("Benchmarking neural net with 1 epoch and 1 5-neuron layer")
# run_benchmark(arr2d([5 5]), 1)
# println("Benchmarking neural net with 1 epoch and 5 5-neuron layers")
# run_benchmark(arr2d([5 5 5 5 5 5]), 1)
# println("Benchmarking neural net with 1 epoch and 10 5-neuron layers")
# run_benchmark(arr2d([5 5 5 5 5 5 5 5 5 5 5]), 1)
# println("Benchmarking neural net with 1 epoch and 50 5-neuron layers")
# run_benchmark(arr2d([5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5]), 1)

#testing matrice-size dependency
println("Benchmarking 10 - 10 - 10 neural net with 1 epoch")
run_benchmark(arr2d([10 10 10]), 1)
println("Benchmarking 20 - 30 - 20 neural net with 1 epoch")
run_benchmark(arr2d([20 30 20]), 1)
println("Benchmarking 20 - 30 - 40 neural net with 1 epoch")
run_benchmark(arr2d([20 30 40]), 1)
println("Benchmarking 40 - 30 - 20 neural net with 1 epoch")
run_benchmark(arr2d([40 30 20]), 1)
