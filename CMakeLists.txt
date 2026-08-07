cmake_minimum_required(VERSION 3.15)
project(Neural VERSION 0.4.2 LANGUAGES CXX)

add_executable(Neural 
    src/main.cpp
    include/Neural.cpp
)

include_directories(Neural
    PRIVATE 
        include/
)
