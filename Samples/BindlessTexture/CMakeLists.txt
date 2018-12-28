include_directories(${sample-required-include-dirs})
add_executable(BindlessTextureSample BindlessTextureSample.cpp)
target_link_libraries(BindlessTextureSample ${sample-required-libraries})