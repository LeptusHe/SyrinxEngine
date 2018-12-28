include_directories(${sample-required-include-dirs})
add_executable(RenderToTargetSample RenderToTargetSample.cpp)
target_link_libraries(RenderToTargetSample ${sample-required-libraries})