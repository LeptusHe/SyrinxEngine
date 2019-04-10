set(include_dirs
        ${third_party_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/Sources
        ${PROJECT_SOURCE_DIR}/Sources/Core
        ${PROJECT_SOURCE_DIR}/Sources/Graphics
        ${PROJECT_SOURCE_DIR}/Sources/Resource
        ${PROJECT_SOURCE_DIR}/Sources/Scene
        ${PROJECT_SOURCE_DIR}/Sources/Pipeline)
set(effect-required-include-dirs ${include_dirs} CACHE INTERNAL "include dirs required by samples")


set(libraries Core Graphics ResourceSystem Scene RenderPipeline glew_s glfw glm pugixml)
set(effect-required-libraries ${libraries} CACHE INTERNAL "libraries required by samples")


#add_subdirectory(Skybox)
add_subdirectory(NormalMap)
add_subdirectory(ToneMapping)

add_custom_target(AllEffects)
add_dependencies(AllEffects
        #SkyboxEffect
        NormalMapEffect
        ToneMappingEffect)
