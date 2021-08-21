include_directories(${sample-required-include-dirs})
add_executable(CubicSplineSample SampleMain.cpp)
target_link_libraries(CubicSplineSample ${sample-required-libraries})