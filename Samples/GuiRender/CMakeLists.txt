include_directories(${sample-required-include-dirs})
add_executable(GuiRenderSample GuiRenderSample.cpp)
target_link_libraries(GuiRenderSample ${sample-required-libraries})