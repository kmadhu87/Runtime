cmake_minimum_required(VERSION 3.15)
project(RuntimeProject CXX)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_executable(runtime
    main.cpp
    src/runtime.cpp
)

