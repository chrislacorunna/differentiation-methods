import Pkg
Pkg.activate(".")


for PATH in ("/global", "/testing", "/manual", "/forward", "/backward/test", "/backward/src")
    MODULE_PATH = string(pwd(), PATH)
    if MODULE_PATH ∉ LOAD_PATH  push!(LOAD_PATH, MODULE_PATH) end
end

using Global, TestTools, ForwardTest, BackwardTest, Dates

test_funs = TestFun[]
push!(test_funs, TestFun("Forward", ForwardTest.neuralnet_test))
push!(test_funs, TestFun("Backward", BackwardTest.neuralnet_test))

#testing sigma activation function
println("-----------------------------------------")
println("Testing Sigma function")

test_cases1 = TestCase[]
test_data1 = Layer[]

Wh = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data1, Layer(Wh, x, σ))
ref_solution_sigm = arr2d([0.9933071490757153; 0.999983298578152])


push!(test_data1, Layer(ref_solution_sigm, mean_squared_loss))
push!(test_cases1, (TestCase(() -> neuralnet_test(test_data1, test_funs))))
run_tests(test_cases1)

#testing linear activation function
println("-----------------------------------------")
println("Testing linear function")

test_cases2 = TestCase[]
test_data2 = Layer[]

Wh = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data2, Layer(Wh, x, linear))
ref_solution_sigm = arr2d([5.; 11.])

push!(test_data2, Layer(ref_solution_sigm, mean_squared_loss))
push!(test_cases2, (TestCase(() -> neuralnet_test(test_data2, test_funs))))
run_tests(test_cases2)

# testing ReLU activation function
println("-----------------------------------------")
println("Testing ReLU function")

test_cases3 = TestCase[]
test_data3 = Layer[]

Wh = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data3, Layer(Wh, x, ReLU))
ref_solution_sigm = arr2d([5.; 11.])

push!(test_data3, Layer(ref_solution_sigm, mean_squared_loss))
push!(test_cases3, (TestCase(() -> neuralnet_test(test_data3, test_funs))))
run_tests(test_cases2)

# testing ReLU activation function
println("-----------------------------------------")
println("Testing Softmax function")

test_cases4 = TestCase[]
test_data4 = Layer[]

Wh = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data4, Layer(Wh, x, softmax))
ref_solution_sigm = arr2d([0.0024726231566348; 0.99752737684337])

push!(test_data4, Layer(ref_solution_sigm, mean_squared_loss))
push!(test_cases4, (TestCase(() -> neuralnet_test(test_data4, test_funs))))
run_tests(test_cases4)

#testing sigma activation function on incorrect result
println("-----------------------------------------")
println("Testing Sigma function on incorrect result")

test_cases5 = TestCase[]
test_data5 = Layer[]

Wh = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data5, Layer(Wh, x, σ))
ref_solution_sigm = arr2d([1.9233071490757153; 1.929983298578152])

push!(test_data5, Layer(ref_solution_sigm, mean_squared_loss))
push!(test_cases5, (TestCase(() -> neuralnet_test(test_data5, test_funs))))
run_tests(test_cases5)

#testing linear activation function on incorrect result
println("-----------------------------------------")
println("Testing linear function on incorrect result")

test_cases6 = TestCase[]
test_data6 = Layer[]

Wh = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data6, Layer(Wh, x, linear))
ref_solution_sigm = arr2d([6.; 12.])

push!(test_data6, Layer(ref_solution_sigm, mean_squared_loss))
push!(test_cases6, (TestCase(() -> neuralnet_test(test_data6, test_funs))))
run_tests(test_cases6)

# testing ReLU activation function on incorrect result
println("-----------------------------------------")
println("Testing ReLU function on incorrect result")

test_cases7 = TestCase[]
test_data7 = Layer[]

Wh = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data7, Layer(Wh, x, ReLU))
ref_solution_sigm = arr2d([6.; 12.])

push!(test_data7, Layer(ref_solution_sigm, mean_squared_loss))
push!(test_cases7, (TestCase(() -> neuralnet_test(test_data7, test_funs))))
run_tests(test_cases7)

# testing ReLU activation function on incorrect result
println("-----------------------------------------")
println("Testing Softmax function on incorrect result")

test_cases8 = TestCase[]
test_data8 = Layer[]
Wh = arr2d([5.; 11.])
x = arr2d([1.])
push!(test_data8, Layer(Wh, x, softmax))
ref_solution = arr2d([1.0024726231566348; 1.99752737684337])

push!(test_data8, Layer(ref_solution_sigm, mean_squared_loss))
push!(test_cases8, (TestCase(() -> neuralnet_test(test_data8, test_funs))))
run_tests(test_cases8)
