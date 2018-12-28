include_directories(${sample-required-include-dirs})
add_executable(UpdateUniformSample UpdateUniformSample.cpp)
target_link_libraries(UpdateUniformSample ${sample-required-libraries})