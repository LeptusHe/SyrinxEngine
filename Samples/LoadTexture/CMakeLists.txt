include_directories(${sample-required-include-dirs})
add_executable(LoadTextureSample LoadTextureSample.cpp)
target_link_libraries(LoadTextureSample ${sample-required-libraries})