cmake_minimum_required(VERSION 3.15)
project(Neural VERSION 1.0 LANGUAGES CXX)

add_executable(Neural src/main.cpp)

include_directories(Neural
    PRIVATE 
        include
)