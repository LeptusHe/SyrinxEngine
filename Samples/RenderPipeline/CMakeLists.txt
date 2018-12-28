include_directories(${sample-required-include-dirs})
add_executable(RenderPipelineSample RenderPipelineSample.cpp)
target_link_libraries(RenderPipelineSample ${sample-required-libraries})