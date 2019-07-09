include_directories(${sample-required-include-dirs})
add_executable(LuaScriptSample LuaScriptSample.cpp)
target_link_libraries(LuaScriptSample ${sample-required-libraries} lua)