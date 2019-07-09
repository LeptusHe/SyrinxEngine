include_directories(${sample-required-include-dirs})
add_executable(SPIRVReflection SPIRVReflectionSample.cpp SyrinxFileIncluder.cpp)
target_link_libraries(SPIRVReflection ${sample-required-libraries} spirv-cross-glsl shaderc glew)
