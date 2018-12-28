include_directories(${sample-required-include-dirs})
add_executable(LoadSceneSample LoadSceneSample.cpp)
target_link_libraries(LoadSceneSample ${sample-required-libraries})