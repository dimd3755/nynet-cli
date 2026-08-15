cmake_minimum_required(VERSION 3.15)
project(nynet-cli VERSION 0.8.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nynet
    src/main.cpp
    src/core/Layer.cpp
    src/core/NeuralNetwork.cpp
    src/core/Neuron.cpp
)

include_directories(nynet
    PRIVATE 
        include/
)
