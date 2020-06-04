cmake_minimum_required(VERSION 3.14)
project(SyrinxEngine)

set(CMAKE_CXX_STANDARD 17)

set(BINARY_DIR ${PROJECT_SOURCE_DIR}/Binary/Release)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BINARY_DIR ${PROJECT_SOURCE_DIR}/Binary/Debug)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})

add_subdirectory(ThirdParty)
add_subdirectory(Sources)
add_subdirectory(UnitTest)
add_subdirectory(Samples)
add_subdirectory(Effects)
add_subdirectory(Tools)
