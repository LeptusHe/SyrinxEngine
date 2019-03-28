set(root_dir ${CMAKE_CURRENT_SOURCE_DIR})

# get include dirs
list(APPEND include_dirs
        ${third_party_INCLUDE_DIRS}
        ${root_dir}
        ${PROJECT_SOURCE_DIR}/Sources/Core
        ${PROJECT_SOURCE_DIR}/Sources/Graphics
        ${PROJECT_SOURCE_DIR}/Sources/Resource)


# add source files into sources
list(APPEND sub_dirs
        ${root_dir}
        ${root_dir}/RenderResource
        ${root_dir}/ResourceManager
        ${root_dir}/ResourceLoader
        ${root_dir}/ResourceSerializer)
foreach(sub_dir ${sub_dirs})
    aux_source_directory(${sub_dir} sources_founded)
    #message(STATUS "find source file in dir[${sub_dir}] : ${sources_founded}")
    list(APPEND sources ${sources_founded})
endforeach()
#message(STATUS "source files: ${sources}")


# build core library
include_directories(${include_dirs})
add_library(ResourceSystem STATIC ${sources})
target_link_libraries(ResourceSystem Core Graphics)