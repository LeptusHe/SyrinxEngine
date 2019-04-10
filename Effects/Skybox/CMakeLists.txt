include_directories(${effect-required-include-dirs})
add_executable(SkyboxEffect SkyboxEffect.cpp)
target_link_libraries(SkyboxEffect ${effect-required-libraries})