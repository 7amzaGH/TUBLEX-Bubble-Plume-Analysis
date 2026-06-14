cmake_minimum_required(VERSION 3.10)

project(tublex_embedded_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tublex_embedded_runtime
    src/main.cpp
    src/tublex_runtime.cpp
)

target_include_directories(tublex_embedded_runtime PRIVATE
    include
)
