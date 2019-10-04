set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(${CMAKE_MODULE_PATH}/ConfigureOptix.cmake)
find_package(OpenGL REQUIRED)

list(APPEND source_dirs ${CMAKE_CURRENT_SOURCE_DIR})
foreach(dir ${source_dirs})
    aux_source_directory(${dir} sources_founded)
    list(APPEND sources ${sources_founded})
endforeach()


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
cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)


add_executable(SyrinxRadiance ${sources} devicePrograms.cu ${embedded_ptx_code})
target_include_directories(SyrinxRadiance PRIVATE ${include_dirs})
target_link_libraries(SyrinxRadiance
    Core
	Scene
    ${OPENGL_gl_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY})
