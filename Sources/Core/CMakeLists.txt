set(root_dir ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND include_dirs ${root_dir} ${third_party_INCLUDE_DIRS})

# add source files into sources
list(APPEND sub_dirs Common Container Exception FileSystem Image Input Logging Math Script Serializer Streaming Time)
foreach(sub_dir ${sub_dirs})
    aux_source_directory(${sub_dir} sources_founded)
    #message(STATUS "find source file in dir[${sub_dir}] : ${sources_founded}")
    list(APPEND sources ${sources_founded})
endforeach()
#message(STATUS "source files: ${sources}")

# build core library
include_directories(${include_dirs})
if (BUILD_SHARED_LIBRARY)
    add_library(Core SHARED ${sources})
else()
    add_library(Core STATIC ${sources})
endif()
target_link_libraries(Core glfw fmt-header-only eigen -lstdc++fs)
