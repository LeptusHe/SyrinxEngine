# get include dirs
set(include_dirs
        ${third_party_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/Sources/
        ${PROJECT_SOURCE_DIR}/Sources/Core
        ${PROJECT_SOURCE_DIR}/Sources/Graphics
        ${PROJECT_SOURCE_DIR}/Sources/Resource)

set(tool-required-include-dirs ${include_dirs} CACHE INTERNAL "include dirs required by tools")


# get libraries
set(libraries Core Graphics ResourceSystem RenderPipeline tinyobjloader glew_s glfw glm)
set(tool-required-libraries ${libraries} CACHE INTERNAL "libraries required by tools")


# build tools
add_subdirectory(SyrinxModelExporter)

add_custom_target(AllTools)
add_dependencies(AllTools SyrinxModelExporter)
