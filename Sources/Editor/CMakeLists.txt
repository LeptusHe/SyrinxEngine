set(root_dir ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND include_dirs
        ${root_dir}
        ${third_party_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/Sources
        ${PROJECT_SOURCE_DIR}/Sources/Core
        ${PROJECT_SOURCE_DIR}/Sources/Graphics
        ${PROJECT_SOURCE_DIR}/Sources/GUI
        ${PROJECT_SOURCE_DIR}/Sources/Pipeline
        ${PROJECT_SOURCE_DIR}/Sources/Resource
        ${PROJECT_SOURCE_DIR}/Sources/Scene)

# add source files into sources
list(APPEND sub_dirs ${root_dir})
foreach(sub_dir ${sub_dirs})
    aux_source_directory(${sub_dir} sources_founded)
    #message(STATUS "find source file in dir[${sub_dir}] : ${sources_founded}")
    list(APPEND sources ${sources_founded})
endforeach()
#message(STATUS "source files: ${sources}")

# build core library
include_directories(${include_dirs})

add_executable(SyrinxEditor ${sources})
target_link_libraries(SyrinxEditor GUI RenderPipeline)
