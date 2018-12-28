aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} sources_founded)

include_directories(${tool-required-include-dirs})
add_executable(SyrinxModelExporter ${sources_founded})
target_link_libraries(SyrinxModelExporter ${tool-required-libraries} assimp)