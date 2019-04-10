include_directories(${effect-required-include-dirs})
add_executable(NormalMapEffect NormalMapEffect.cpp)
target_link_libraries(NormalMapEffect ${effect-required-libraries})