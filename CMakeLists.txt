cmake_minimum_required(VERSION 3.15)
project(Neural VERSION 0.6.4 LANGUAGES CXX)

add_executable(Neural 
    src/main.cpp
    src/core/Layer.cpp
    src/core/NeuralNetwork.cpp
    src/core/Neuron.cpp
)

include_directories(Neural
    PRIVATE 
        include/
)
