# get include dirs
set(include_dirs
        ${third_party_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/Sources
        ${PROJECT_SOURCE_DIR}/Sources/Core
        ${PROJECT_SOURCE_DIR}/Sources/Graphics
        ${PROJECT_SOURCE_DIR}/Sources/Resource
        ${PROJECT_SOURCE_DIR}/Sources/Scene
        ${PROJECT_SOURCE_DIR}/Sources/Pipeline
        ${PROJECT_SOURCE_DIR}/Sources/GUI)

set(sample-required-include-dirs ${include_dirs} CACHE INTERNAL "include dirs required by samples")


# get libraries
set(libraries RenderPipeline GUI)
set(sample-required-libraries ${libraries} CACHE INTERNAL "libraries required by samples")


# build samples
add_subdirectory(CameraMotion)
add_subdirectory(CreateWindow)
add_subdirectory(DrawTriangle)
add_subdirectory(UpdateUniform)
add_subdirectory(LoadMesh)
add_subdirectory(LoadModel)
add_subdirectory(LoadMaterial)
add_subdirectory(LoadTexture)
add_subdirectory(LuaScript)
add_subdirectory(MotionController)
add_subdirectory(ParametricCurve)
add_subdirectory(RenderToTarget)
add_subdirectory(LoadScene)
add_subdirectory(RenderPipeline)
add_subdirectory(TextureMipMap)
add_subdirectory(SPIRVReflection)
add_subdirectory(GuiRender)


add_custom_target(AllSamples)
add_dependencies(AllSamples
        CameraMotionSample
        CreateWindowSample
        DrawTriangleSample
        UpdateUniformSample
        LoadTextureSample
        LoadMeshSample
        LoadModelSample
        LoadMaterialSample
        LuaScriptSample
        MotionControllerSample
        RenderToTargetSample
        RenderPipelineSample
        LoadSceneSample
        TextureMipMapSample)
