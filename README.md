# Neural Network
![Static Badge](https://img.shields.io/badge/Version-v0.6.4-yellow)
![Static Badge](https://img.shields.io/badge/License-Apache%202.0-yellow)
![Static Badge](https://img.shields.io/badge/Build-passing-green)


A lightweight, dependency-free C++ Neural Network built entirely from scratch using only the standard library (`std`). 

This project was developed as a hands-on educational exercise to deeply understand and implement the underlying mathematics of artificial neural networks—specifically forward propagation, backpropagation, and activation functions—without relying on heavy frameworks like PyTorch or TensorFlow.

# Requirements
1. (*Optional*) [**Git**](https://git-scm.com/install/) for cloning the project.
2. [CMake](https://cmake.org/download/) for building the project.
3. C++ Compiler (eg. GNU, Clang, MSVC) for compiling the project into binary.

# Set up
First you need to clone the project with git, (or just download the [.zip](https://github.com/dimd3755/NeuralNetwork/archive/refs/heads/master.zip) file):

    git clone https://github.com/dimd3755/NeuralNetwork


Once you are inside the repository, use cmake to build the project:

    mkdir build

    cd build

    cmake ..

    cmake --build .

and you will find the executable.

# Features
It offers a 'NeuralNetwork' which is the highest level implementation of 'Layer' and 'Neuron' structs.

The class mentioned above has 3 Layers which all 3 have customisable sizes, and they are:

1. **Input Layer**: which is a vector holding values with a constant type of 'double', so incase you want to input something in the network you would have to cast it to double.

2. **Middle Layer**: which's main purpose is data and feature extraction, storage, and also helps the network to solve more complex calculations. This layer uses the ReLU activation function by default.

3. **Output Layer**: which is the result of the calculation, it is essential for giving feedback back to the network for adjustments. This layer uses the Sigmoid activation function.

# Usage 
It is pretty easy to use NeuralNetwork class as it only has 4 functions, with them being:

* **`NeuralNetwork()`** being the constructor, the parameters it takes is **`const size_t& input_num_neurons, const std::vector<size_t>& hidden_neuron_sizes, const size_t& output_num_neurons, const activate_type& hidden_func=activate_type::RELU, const activate_type& output_func=activate_type::SIGMOID, const double& lr=0.001`**, where the 1st parameter is the neuron amount of input layer, the next is the hidden layer vector which is how many neurons will each layer have and how many hidden layers are, the 3rd is the amount of neurons for the output, next we have the activation type for hidden layer, and then the activation type for the output layer, and lastly the learning rate, which is 0.001 by default.

* **`forward()`** which takes as parameter **`const std::vector<double>& inputs`** being the input vector to calculate the output.

* **`train()`** which uses the **'forward'** function to calculate the output then compare with the **target** to make adjustments, and its parameters are **`const std::vector<double>& inputs, const std::vector<double>& target`**, where the 'inputs' is the training data, and the 'target' is the desired output.

* **`get_output()`** which returns the output of the `train()` or `forward()` functions, it takes no parameters, and it's value is returned as read-only.

Here is an example of training the model to do the XOR operation:

``` cpp
#include "Neural.hpp"
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    // 2 Neurons for inputs, 4x2 hidden layers, 1 output, RELU activation type for hidden, Sigmoid for output
    // learning rate is set 0.01
    NeuralNetwork nn(2, {4, 4}, 1, activate_type::RELU, activate_type::SIGMOID, 0.01);

    std::vector<std::vector<double>> x = 
    {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
    };

    std::vector<std::vector<double>> y =
    {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Training
    for (size_t i = 0; i < 10000; ++i) {
        for (size_t j = 0; j < x.size(); ++j) {
            nn.train(x[j], y[j]);
        }
    }

    // Testing 
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        nn.forward(inputs[i]);
        std::vector<double> output = nn.get_output();
        std::cout << "Prediction of {" << inputs[i][0] << ", " << inputs[i][1] << "}: ";
        for (auto& prediction: output) {
            std::cout << prediction << '\n';
        }
    }
    return 0;
}

```

# Up-coming features
Since this current repository is lacking many features to make it achieve its purpose as a tool for analysis and experiments, expect some features on next updates. Some of the features are:

* More activation functions: there will be more than just ReLU, Sigmoid, and Tanh, functions like GeLU, Leaky ReLU, and Softmax are upcoming.

* Training data file support: which will increase the portability of the tool making it able to hold different training sets

* Output data file support: which will make it easier to handle large results in a single file.

# Summary
This tool will be pretty useful in the future for solving many problems and making predictions, it offers high level functions, while it also offers portability for different purposes.
