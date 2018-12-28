include_directories(${sample-required-include-dirs})
add_executable(LoadModelSample LoadModelSample.cpp)
target_link_libraries(LoadModelSample ${sample-required-libraries})