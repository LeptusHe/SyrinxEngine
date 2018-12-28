# get source files
list(APPEND dirs TestDouble Core Graphics ResourceSystem Scene)
foreach(dir ${dirs})
    aux_source_directory(${dir} sources-founded)
    list(APPEND sources ${sources-founded})
    #message(WARNING "find source files in dir[${dir}]: [${sources-founded}]")
endforeach()


# set include directories
set(source-root-dir ${PROJECT_SOURCE_DIR}/Sources)
list(APPEND include-dirs
        ${third_party_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${source-root-dir}
        ${source-root-dir}/Core
        ${source-root-dir}/Graphics
        ${source-root-dir}/ResourceSystem
        ${source-root-dir}/Scene)


# set libraries
list(APPEND thirdparty-libraries glew_s pugixml)
list(APPEND libraries gmock Core Graphics ResourceSystem Scene RenderPipeline ${thirdparty-libraries})


# build
include_directories(${include-dirs})
add_executable(SyrinxUnitTest ${sources} GTestMain.cpp)
target_link_libraries(SyrinxUnitTest ${libraries})