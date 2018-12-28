include_directories(${sample-required-include-dirs})
add_executable(LoadMaterialSample LoadMaterialSample.cpp)
target_link_libraries(LoadMaterialSample ${sample-required-libraries})