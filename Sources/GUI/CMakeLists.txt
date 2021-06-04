set(root_dir ${CMAKE_CURRENT_SOURCE_DIR})

# get include dirs
list(APPEND include_dirs
        ${third_party_INCLUDE_DIRS}
        ${root_dir}
        ${PROJECT_SOURCE_DIR}/Sources
        ${PROJECT_SOURCE_DIR}/Sources/Core
        ${PROJECT_SOURCE_DIR}/Sources/Graphics
        ${PROJECT_SOURCE_DIR}/Sources/Resource
        ${PROJECT_SOURCE_DIR}/Sources/Pipeline
)


# add source files into sources
list(APPEND sub_dirs ${root_dir} Widgets)
foreach(sub_dir ${sub_dirs})
    aux_source_directory(${sub_dir} sources_founded)
    #message(STATUS "find source file in dir[${sub_dir}] : ${sources_founded}")
    list(APPEND sources ${sources_founded})
endforeach()
#message(STATUS "source files: ${sources}")


# build core library
include_directories(${include_dirs})
if (BUILD_SHARED_LIBRARY)
    add_library(GUI SHARED ${sources})
else()
    add_library(GUI STATIC ${sources})
endif()
target_link_libraries(GUI Graphics imgui implot)
