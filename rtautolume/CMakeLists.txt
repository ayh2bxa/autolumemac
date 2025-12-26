cmake_minimum_required(VERSION 3.22)

set(PROJECT_NAME "AutolumeJUCE")

project(${PROJECT_NAME} VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR}/plugin)

# Platform-specific compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set JUCE directory (allow override via environment variable or CMake variable)
if(NOT DEFINED JUCE_DIR)
    if(WIN32)
        set(JUCE_DIR "$ENV{USERPROFILE}/Documents/JUCE")
    else()
        set(JUCE_DIR "$ENV{HOME}/Documents/JUCE")
    endif()
endif()

add_subdirectory(${JUCE_DIR} ${CMAKE_BINARY_DIR}/juce)
add_subdirectory(plugin)