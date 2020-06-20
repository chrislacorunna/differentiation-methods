import Pkg
Pkg.activate(".")

for PATH in ("/global", "/testing", "/manual", "/forward", "/backward/src")
    MODULE_PATH = string(pwd(), PATH)
    if MODULE_PATH ∉ LOAD_PATH  push!(LOAD_PATH, MODULE_PATH) end
end

function get_rand_activation()
    return rand([linear, ReLU, σ, softmax])
end
# function for testing random values. should be a tab where 1st element is
# x vector and next ones are neurons and the last one is output
function add_benchmark(name, neuron_tab, epochs)
    test_data = Layer[]
    W = arr2d(randn(neuron_tab[2], neuron_tab[1]))
    x = arr2d(randn(neuron_tab[1], 1))
    push!(test_data, Layer(W, x, σ))

    for i = 3:size(neuron_tab,2)
        Wn = arr2d(randn(neuron_tab[i], neuron_tab[i-1]))
        push!(test_data, Layer(Wn, get_rand_activation()))
    end

    push!(test_cases, (TestCase(name, () -> neuralnet_test(test_data, test_funs, epochs, benchmark=true))))
end


using Global, TestTools, ForwardTest, BackwardTest

test_cases = TestCase[]
test_funs = TestFun[]
push!(test_funs, TestFun("Forward", ForwardTest.neuralnet_test))
push!(test_funs, TestFun("Backward", BackwardTest.neuralnet_test))

# testing matrice-size dependency
add_benchmark("Benchmarking 10 - 10 - 10 neural net with 1 epoch", arr2d([10 10 10]), 1)
add_benchmark("Benchmarking 20 - 30 - 20 neural net with 1 epoch", arr2d([20 30 20]), 1)
add_benchmark("Benchmarking 20 - 30 - 40 neural net with 1 epoch", arr2d([20 30 40]), 1)
add_benchmark("Benchmarking 40 - 30 - 20 neural net with 1 epoch", arr2d([40 30 20]), 1)

#testing resource-dependency for epochs
add_benchmark("Benchmarking 2 - 3 - 2 neural net with 1 epoch", arr2d([2 3 2]), 1)
add_benchmark("Benchmarking 2 - 3 - 2 neural net with 4 epochs", arr2d([2 3 2]), 4)
add_benchmark("Benchmarking 2 - 3 - 2 neural net with 16 epochs", arr2d([2 3 2]), 16)
add_benchmark("Benchmarking 2 - 3 - 2 neural net with 64 epochs", arr2d([2 3 2]), 64)
add_benchmark("Benchmarking 2 - 3 - 2 neural net with 256 epochs", arr2d([2 3 2]), 256)

#testing layer-number dependency
add_benchmark("Benchmarking neural net with 1 epoch and 1 5-neuron layer",
    arr2d([5 5]), 1)
add_benchmark("Benchmarking neural net with 1 epoch and 5 5-neuron layers",
    arr2d([5 5 5 5 5 5]), 1)
add_benchmark("Benchmarking neural net with 1 epoch and 10 5-neuron layers",
    arr2d(5*ones(Int64, 1, 11)), 1)
add_benchmark("Benchmarking neural net with 1 epoch and 50 5-neuron layers",
    arr2d(5*ones(Int64, 1, 51)), 1)
add_benchmark("Benchmarking neural net with 1 epoch and 100 5-neuron layers",
    arr2d(5*ones(Int64, 1, 101)), 1)

######################################
run_tests(test_cases)
