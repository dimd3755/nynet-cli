# 🧠 Neural Network
![Static Badge](https://img.shields.io/badge/Version-v0.3.0-yellow)
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

# ⚙ Features
It offers a 'NeuralNetwork' which is the highest level implementation of 'Layer' and 'Neuron' structs.

The class mentioned above has 3 Layers which all 3 have customisable sizes, and they are:

1. **Input Layer**: which is a vector holding values with a constant type of 'double', so incase you want to input something in the network you would have to cast it to double.

2. **Middle Layer**: which's main purpose is data and feature extraction, storage, and also helps the network to solve more complex calculations. This layer uses the ReLU activation function.

3. **Output Layer**: which is the result of the calculation, it is essential for giving feedback back to the network for adjustments. This layer uses the Sigmoid activation function.

# ▶ Usage 
It is pretty easy to use NeuralNetwork class as it only has 4 functions, with them being:

* **'NeuralNetwork()'** being the constructor, the parameters it takes is **`const size_t& num_inputs, const size_t& hidden_num_inputs, const size_t& num_outputs, const double& lr = 0.1`**, which is basically the size of the 1st layer (input), the 2nd layer (middle), the 3rd layer (output), and the learning rate (which is optional since 0.1 is set by default).

* **'forward()'** which takes as parameter **`const std::vector<double>& inputs`** being the input vector to calculate the output.

* **`train()`** which uses the **'forward'** function to calculate the output then compare with the **target** to make adjustments, and its parameters are **`const std::vector<double>& target, const std::vector<double>& inputs`**, where the 'inputs' is the training data, and the 'target' is the desired output.

* **`get_output()`** which returns the output of the `train()` or `forward()` functions, it takes no parameters, and it's value is returned as read-only.

Here is a quick example of training the model to do the XOR operation:

``` cpp
#include "Neural.hpp"
#include <vector>

int main() {
    // 2 Inputs, 3 Neurons as middle layer, 1 output
    NeuralNetwork nn(2, 3, 1, 0.1);

    std::vector<double> inputs = {1.0, 0.0};
    std::vector<double> target = {1.0};

    nn.train(target, inputs); //Training the model

    std::vector<double> output = nn.get_output(); //Gets training output
    return 0;
}
```


# ⏳Up-coming features
Since this current repository is lacking many features to make it achieve its purpose as a tool for analysis and experiments, expect some features on next updates. Some of the features are:

* Customisable amount of layers: which would be essential to make it calculate more complex problems.

* Saved adjustments and settings: it will read a file input and will adjust the bias and weights of the neurons, aswell as the sizes and length of the layers, as instructed to continue the training from where you left.

* Training data file support: which will increase the portability of the tool making it able to hold different training sets

* Output data file support: which will make it easier to handle large results in a single file.

# ⏩ Summary
This tool will be pretty useful in the future for solving many problems and making predictions, it offers high level functions, while it also offers portability for different purposes.
