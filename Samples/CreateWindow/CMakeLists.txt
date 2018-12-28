include_directories(${sample-required-include-dirs})
add_executable(CreateWindowSample CreateWindowSample.cpp)
target_link_libraries(CreateWindowSample ${sample-required-libraries})