include_directories(${sample-required-include-dirs})
add_executable(NormalMapSample NormalMapSample.cpp)
target_link_libraries(NormalMapSample ${sample-required-libraries})