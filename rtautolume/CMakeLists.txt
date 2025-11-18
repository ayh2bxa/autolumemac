cmake_minimum_required(VERSION 3.22)

set(PROJECT_NAME "AutolumeMac")

project(${PROJECT_NAME} VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR}/plugin)

add_compile_options(-Wall -Wextra -Wpedantic)

set(JUCE_DIR "$ENV{HOME}/Documents/JUCE")
add_subdirectory(${JUCE_DIR} ${CMAKE_BINARY_DIR}/juce)
add_subdirectory(plugin)