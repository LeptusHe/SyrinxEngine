include_directories(${effect-required-include-dirs})
add_executable(ToneMappingEffect ToneMappingEffect.cpp)
target_link_libraries(ToneMappingEffect ${effect-required-libraries})