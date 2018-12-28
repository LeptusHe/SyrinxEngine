include_directories(${sample-required-include-dirs})
add_executable(DrawTriangleSample DrawTriangleSample.cpp)
target_link_libraries(DrawTriangleSample ${sample-required-libraries})