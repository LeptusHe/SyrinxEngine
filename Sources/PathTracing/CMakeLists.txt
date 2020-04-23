set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

set(USER_OPTIX_INSTALL_DIR "D:/ProgramFiles/NVIDIACorporation/OptiXSDK7.0.0")

include(${CMAKE_MODULE_PATH}/ConfigureOptix.cmake)
find_package(OpenGL REQUIRED)

list(APPEND source_dirs ${CMAKE_CURRENT_SOURCE_DIR})
foreach(dir ${source_dirs})
    aux_source_directory(${dir} sources_founded)
    list(APPEND sources ${sources_founded})
endforeach()

if (NOT ${searched_OptiX_INSTALL_DIR-NOTFOUND})
	message(ERROR "fail to find Optix include dir")
else()
	message(STATUS "Optix Include Dir: " ${OptiX_INCLUDE})
endif()

list(APPEND include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/Sources
	${PROJECT_SOURCE_DIR}/Sources/Core
	${PROJECT_SOURCE_DIR}/Sources/Graphics
	${PROJECT_SOURCE_DIR}/Sources/Pipeline
	${PROJECT_SOURCE_DIR}/Sources/Resource
	${PROJECT_SOURCE_DIR}/Sources/Scene
	${OptiX_INCLUDE}
	$ENV{CUDA_PATH}/include/
	${third_party_INCLUDE_DIRS})
include_directories(${include_dirs})

list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}"--gpu-architecture=sm_50" "-rdc=true" "--use_fast_math")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "cuda compile flags")
cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)


add_executable(SyrinxRadiance ${sources} devicePrograms.cu ${embedded_ptx_code})
target_include_directories(SyrinxRadiance PRIVATE ${include_dirs})
target_link_libraries(SyrinxRadiance
    Core
	Scene
    ${OPENGL_gl_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY})
