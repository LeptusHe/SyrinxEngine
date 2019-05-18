set(root_dir ${CMAKE_CURRENT_SOURCE_DIR})

# build third party libraries
# build assimp
set(BUILD_SHARED_LIBS OFF CACHE BOOL "build static library")
set(ASSIMP_BUILD_ASSIMP_TOOLS FALSE CACHE BOOL "disable build assimp tools")
set(ASSIMP_BUILD_TESTS FALSE CACHE BOOL "disable build tests")
set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT FALSE CACHE BOOL "disable build all importer")
set(ASSIMP_BUILD_OBJ_IMPORTER TRUE CACHE BOOL "enable obj importer")
set(ASSIMP_BUILD_STL_IMPORTER TRUE CACHE BOOL "enable stl importer")
set(ASSIMP_BUILD_PLY_IMPORTER TRUE CACHE BOOL "enable ply importer")
set(ASSIMP_BUILD_FBX_IMPORTER TRUE CACHE BOOL "enable fbx importer")
#add_subdirectory(assimp)

# build cxxopts
add_subdirectory(cxxopts)

# build entityx
set(ENTITYX_BUILD_TESTING FALSE CACHE BOOL "disable build test")
set(ENTITYX_DT_TYPE float CACHE STRING "use float type for delta time")
set(ENTITYX_BUILD_SHARED  FALSE CACHE BOOL "build static library")
add_subdirectory(entityx)

# build fmt library
add_subdirectory(fmt)

# build gmock library
set(BUILD_GMOCK ON CACHE BOOL "build gmock")
set(INSTALL_GTEST OFF CACHE BOOL "disable to install gtest")
add_subdirectory(googletest)

# build glfw library
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "disable build example for glfw")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "disable build test for glfw")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "disable build test for glfw")
set(GLFW_INSTALL OFF CACHE BOOL "disable build test for glfw")
add_subdirectory(glfw)

# build glew library
set(BUILD_UTILS OFF CACHE BOOL "disable build utils for glew")
add_subdirectory(glew/build/cmake)

# build glm library
set(GLM_TEST_ENABLE OFF CACHE BOOL "disable build test")
add_subdirectory(glm)

# build imgui
add_subdirectory(imgui)

# build lua
aux_source_directory(lua lua-sources)
add_library(lua ${lua-sources})

# build pugixml
add_subdirectory(pugixml)

# build SPIRV-Cross
set(SPIRV_CROSS_SHARED OFF CACHE BOOL "Build the C API as a single shared library.")
set(SPIRV_CROSS_STATIC ON CACHE BOOL "Build the C and C++ API as static libraries.")
set(SPIRV_CROSS_CLI OFF CACHE BOOL "Build the CLI binary. Requires SPIRV_CROSS_STATIC.")
set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "Enable SPIRV-Cross tests.")
set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "Enable HLSL target support.")
set(SPIRV_CROSS_ENABLE_MSL OFF CACHE BOOL "Enable MSL target support.")
set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "Enable C++ target support.")
set(SPIRV_CROSS_ENABLE_C_API ON CACHE BOOL "Enable C API wrapper support in static library.")
add_subdirectory(SPIRV-Cross)

# build tinyobjloader library
add_subdirectory(tinyobjloader)

# get include dirs for libraries
set(assimp-headers ${root_dir}/assimp/include)
set(better-enums-headers ${root_dir})
set(cxxopts-headers ${root_dir}/cxxopts/include)
set(entityx-headers ${root_dir}/entityx)
set(fmt-headers ${root_dir}/fmt/include)
set(gmock-headers ${root_dir}/googletest/googlemock/include)
set(glew-headers ${root_dir}/glew/include)
set(glfw-headers ${root_dir}/glfw/include)
set(glm-headers ${root_dir}/glm)
set(imgui-headers ${root_dir})
set(json-headers ${root_dir}/json/single_include)
set(lua-headers ${root_dir}/lua)
set(pugixml-headers ${root_dir}/pugixml/src)
set(SPIRV-Cross-header ${root_dir}/SPIRV-Cross)
set(tinyobjloader-headers ${root_dir}/tinyobjloader)

# set headers for third party
list(APPEND third_party_headers
        ${assimp-headers}
        ${better-enums-headers}
        ${cxxopts-headers}
        ${entityx-headers}
        ${fmt-headers}
        ${gmock-headers}
        ${glew-headers}
        ${glfw-headers}
        ${glm-headers}
        ${imgui-headers}
        ${json-headers}
        ${lua-headers}
        ${pugixml-headers}
        ${SPIRV-Cross-header}
        ${tinyobjloader-headers})
set(third_party_INCLUDE_DIRS ${third_party_headers} CACHE INTERNAL "include directories for third party libraries")