include_directories(${sample-required-include-dirs})
add_executable(LoadMeshSample LoadMeshSample.cpp)
target_link_libraries(LoadMeshSample ${sample-required-libraries})