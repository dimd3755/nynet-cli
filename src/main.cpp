#include "Neural.hpp"
#include <cstddef>
#include <vector>
#include <iostream>

template <class T>
void printVec(const std::vector<T>& vec);

int main() {
    std::cout << "Start" << std::endl;
    
    NeuralNetwork nn(2, 3, 1);

    std::vector<std::vector<double>> training_data = {
        {1.0, 0.0},
        {0.0, 0.0}
    };

    std::vector<std::vector<double>> target_data = {
        {1.0},
        {0.0}
    };

    for (size_t i = 0; i < 20; ++i) {
        for (size_t j = 0; j < training_data.size(); ++j) {
            nn.train(target_data[j], training_data[j]);
        }
    }
    nn.forward({1.0, 1.0});

    printVec<double>(nn.get_output());
    return 0;
}

template <class T>
void printVec(const std::vector<T>& vec) {
    std::cout << '[';
    for (auto& c: vec) {
        if (c == vec[0]) {
            std::cout << c;
        } else {
            std::cout << ", " << c;
        }
    }
    std::cout << ']' << '\n';
}