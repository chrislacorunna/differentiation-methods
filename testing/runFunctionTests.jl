import Pkg
Pkg.activate(".")

for PATH in ("/global", "/testing", "/manual", "/forward", "/backward/src")
    MODULE_PATH = string(pwd(), PATH)
    if MODULE_PATH ∉ LOAD_PATH  push!(LOAD_PATH, MODULE_PATH) end
end

using Global, TestTools, ForwardTest, BackwardTest, Dates

test_cases = TestCase[]
test_funs = TestFun[]
push!(test_funs, TestFun("Forward", ForwardTest.neuralnet_test))
push!(test_funs, TestFun("Backward", BackwardTest.neuralnet_test))

#testing sigma activation function
test_data1 = Layer[]

W = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data1, Layer(W, x, σ))
ref_solution = arr2d([0.9933071490757151; 0.999983298578152])

push!(test_data1, Layer(ref_solution, mean_squared_loss))
push!(test_cases, (TestCase("Testing sigma function",
    () -> neuralnet_test(test_data1, test_funs, 1))))

#testing linear activation function
test_data2 = Layer[]

W = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data2, Layer(W, x, linear))
ref_solution = arr2d([5.; 11.])

push!(test_data2, Layer(ref_solution, mean_squared_loss))
push!(test_cases, (TestCase("Testing linear function",
    () -> neuralnet_test(test_data2, test_funs, 1))))

# testing ReLU activation function
test_data3 = Layer[]

W = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data3, Layer(W, x, ReLU))
ref_solution = arr2d([5.; 11.])

push!(test_data3, Layer(ref_solution, mean_squared_loss))
push!(test_cases, (TestCase("Testing ReLU function",
    () -> neuralnet_test(test_data3, test_funs, 1))))

# testing softmax activation function
test_data4 = Layer[]

W = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data4, Layer(W, x, softmax))
ref_solution = arr2d([0.0024726231566348; 0.99752737684337])

push!(test_data4, Layer(ref_solution, mean_squared_loss))
push!(test_cases, (TestCase("Testing softmax function",
    () -> neuralnet_test(test_data4, test_funs, 1))))

#testing sigma activation function on incorrect result
test_data5 = Layer[]

W = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data5, Layer(W, x, σ))
ref_solution_plus_one = arr2d([1.9233071490757153; 1.929983298578152])

push!(test_data5, Layer(ref_solution_plus_one, mean_squared_loss))
push!(test_cases, (TestCase("Testing sigma function on incorrect result",
    () -> neuralnet_test(test_data5, test_funs, 1))))

#testing linear activation function on incorrect result
test_data6 = Layer[]

W = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data6, Layer(W, x, linear))
ref_solution_plus_one = arr2d([6.; 12.])

push!(test_data6, Layer(ref_solution_plus_one, mean_squared_loss))
push!(test_cases, (TestCase("Testing linear function on incorrect result",
    () -> neuralnet_test(test_data6, test_funs, 1))))

# testing ReLU activation function on incorrect result
test_data7 = Layer[]

W = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data7, Layer(W, x, ReLU))
ref_solution_plus_one = arr2d([6.; 12.])

push!(test_data7, Layer(ref_solution_plus_one, mean_squared_loss))
push!(test_cases, (TestCase("Testing ReLU function on incorrect result",
    () -> neuralnet_test(test_data7, test_funs, 1))))

# testing ReLU activation function on incorrect result
test_data8 = Layer[]
W = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data8, Layer(W, x, softmax))
ref_solution_plus_one = arr2d([1.0024726231566348; 1.99752737684337])

push!(test_data8, Layer(ref_solution_plus_one, mean_squared_loss))
push!(test_cases, (TestCase("Testing Softmax function on incorrect result",
    () -> neuralnet_test(test_data8, test_funs, 1))))
################################
run_tests(test_cases)
