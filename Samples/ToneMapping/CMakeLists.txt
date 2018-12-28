include_directories(${sample-required-include-dirs})
add_executable(ToneMappingSample ToneMappingSample.cpp)
target_link_libraries(ToneMappingSample ${sample-required-libraries})