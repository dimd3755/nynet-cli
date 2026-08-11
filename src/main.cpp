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