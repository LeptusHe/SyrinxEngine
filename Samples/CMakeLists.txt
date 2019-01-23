# get include dirs
set(include_dirs
        ${third_party_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/Sources
        ${PROJECT_SOURCE_DIR}/Sources/Core
        ${PROJECT_SOURCE_DIR}/Sources/Graphics
        ${PROJECT_SOURCE_DIR}/Sources/ResourceSystem
        ${PROJECT_SOURCE_DIR}/Sources/Scene
        ${PROJECT_SOURCE_DIR}/Sources/RenderPipeline)

set(sample-required-include-dirs ${include_dirs} CACHE INTERNAL "include dirs required by samples")


# get libraries
set(libraries Core Graphics ResourceSystem Scene RenderPipeline glew_s glfw glm pugixml)
set(sample-required-libraries ${libraries} CACHE INTERNAL "libraries required by samples")


# build samples
add_subdirectory(CreateWindow)
add_subdirectory(DrawTriangle)
add_subdirectory(UpdateUniform)
add_subdirectory(LoadTexture)
add_subdirectory(LoadMesh)
add_subdirectory(LoadModel)
add_subdirectory(LoadMaterial)
add_subdirectory(MotionController)
add_subdirectory(RenderToTarget)
add_subdirectory(Skybox)
add_subdirectory(LoadScene)
add_subdirectory(RenderPipeline)
add_subdirectory(TextureMipMap)
add_subdirectory(ToneMapping)
add_subdirectory(BindlessTexture)
add_subdirectory(NormalMap)


add_custom_target(AllSamples)
add_dependencies(AllSamples
        CreateWindowSample
        DrawTriangleSample
        UpdateUniformSample
        LoadTextureSample
        LoadMeshSample
        LoadModelSample
        LoadMaterialSample
        MotionControllerSample
        RenderToTargetSample
        RenderPipelineSample
        SkyboxSample
        LoadSceneSample
        TextureMipMapSample
        BindlessTextureSample)
