include_directories(${sample-required-include-dirs})
add_executable(ParametricCurveSample ParametricCurveSample.cpp)
target_link_libraries(ParametricCurveSample ${sample-required-libraries})