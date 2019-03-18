include_directories(${sample-required-include-dirs})
add_executable(CameraMotionSample CameraMotionSample.cpp)
target_link_libraries(CameraMotionSample ${sample-required-libraries})