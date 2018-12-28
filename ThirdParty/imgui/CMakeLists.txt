list(APPEND include-dirs
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/ThirdParty/glew/include
        ${PROJECT_SOURCE_DIR}/ThirdParty/glfw/include)
include_directories(${include-dirs})
list(APPEND sources imgui.cpp imgui_draw.cpp imgui_widgets.cpp examples/imgui_impl_glfw.cpp examples/imgui_impl_opengl3.cpp)
add_library(imgui ${sources})
target_link_libraries(imgui glew_s)