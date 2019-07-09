cmake_minimum_required(VERSION 3.14)
project(SyrinxEngine)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(ThirdParty)
add_subdirectory(Sources)
add_subdirectory(UnitTest)
add_subdirectory(Samples)
add_subdirectory(Effects)
add_subdirectory(Tools)
